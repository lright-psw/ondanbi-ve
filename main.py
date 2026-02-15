import copy
import json
import re
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
import sounddevice as sd
import soundfile as sf
from PyQt6.QtCore import QSize, Qt, QTimer
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QStyle,
    QTabWidget,
)


# ============================================================
# 데이터 모델 (오디오 이펙트 파라미터)
# ============================================================
@dataclass
class EffectParams:
    """실시간 오디오 처리에 사용하는 이펙트/게인 파라미터 묶음"""

    input_gain: float = 1.0
    output_gain: float = 1.0
    distortion_drive: float = 1.0
    reverb_wet: float = 0.0
    delay_wet: float = 0.0
    delay_time_ms: float = 280.0
    delay_feedback: float = 0.35
    echo_wet: float = 0.0
    echo_time_ms: float = 650.0
    echo_feedback: float = 0.3


class AudioEngine:
    """실시간 오디오 스트림 처리, 녹음, 이펙트 적용을 담당하는 엔진"""

    # -------- 초기화/장치 설정 --------
    def __init__(self) -> None:
        """오디오 엔진의 기본 상태와 내부 버퍼를 초기화."""
        self.lock = threading.Lock()
        self.params = EffectParams()
        self.stream: Optional[sd.Stream] = None
        self.sample_rate = 48000
        # UI 이벤트가 많은 상황에서도 끊김을 줄이기 위해 버퍼를 조금 크게 둔다.
        self.block_size = 2048
        self.input_channels = 1
        self.output_channels = 1
        self.input_device: Optional[int] = None
        self.output_device: Optional[int] = None

        self.latest_output = np.zeros(self.block_size, dtype=np.float32)
        self.last_stream_status = ""

        self.is_recording = False
        self.recorded_chunks: List[np.ndarray] = []
        # 기본값: 녹음 중에는 스피커로 내보내지 않고 파일에만 저장.
        self.record_output_mode = "mute_while_recording"
        self.active_record_output_mode = self.record_output_mode

        self.reverb_tap_seconds = (0.013, 0.017, 0.019, 0.023)
        self.reverb_tap_gains = np.array([0.45, 0.33, 0.24, 0.18], dtype=np.float32)

        self._reset_effect_buffers()

    def _reset_effect_buffers(self) -> None:
        """샘플레이트 기준으로 리버브/딜레이/에코 버퍼를 재생성"""
        self.max_reverb_samples = max(1, int(0.08 * self.sample_rate))
        self.reverb_buffer = np.zeros(self.max_reverb_samples, dtype=np.float32)
        self.reverb_idx = 0
        self.reverb_taps = [
            max(1, min(int(sec * self.sample_rate), self.max_reverb_samples - 1))
            for sec in self.reverb_tap_seconds
        ]

        self.max_delay_samples = max(2, int(3.0 * self.sample_rate))
        self.delay_buffer = np.zeros(self.max_delay_samples, dtype=np.float32)
        self.delay_idx = 0

        self.max_echo_samples = max(2, int(4.0 * self.sample_rate))
        self.echo_buffer = np.zeros(self.max_echo_samples, dtype=np.float32)
        self.echo_idx = 0

    @staticmethod
    def list_devices() -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
        """시스템에서 사용 가능한 입력/출력 오디오 장치 목록을 반환"""
        inputs: List[Tuple[int, str]] = []
        outputs: List[Tuple[int, str]] = []
        for idx, dev in enumerate(sd.query_devices()):
            name = dev["name"].strip()
            if dev["max_input_channels"] > 0:
                inputs.append((idx, name))
            if dev["max_output_channels"] > 0:
                outputs.append((idx, name))
        return inputs, outputs

    def set_devices(self, input_device: int, output_device: int) -> None:
        """현재 엔진에서 사용할 입력/출력 장치 인덱스를 설정"""
        with self.lock:
            self.input_device = input_device
            self.output_device = output_device

    def set_param(self, key: str, value: float) -> None:
        """슬라이더에서 변경된 파라미터 값을 엔진에 반영"""
        with self.lock:
            setattr(self.params, key, value)

    def set_record_output_mode(self, mode: str) -> None:
        """녹음 중 출력 모드(always / mute_while_recording)를 설정"""
        if mode not in {"always", "mute_while_recording"}:
            return
        with self.lock:
            self.record_output_mode = mode
            if not self.is_recording:
                # 미녹음 상태에서는 다음 녹음 세션 모드도 즉시 동기화한다.
                self.active_record_output_mode = mode

    # -------- 스트림 시작/중지 --------
    def _find_compatible_stream_config(self) -> Tuple[int, int]:
        """선택한 장치에서 동작 가능한 샘플레이트/출력 채널 조합을 탐색"""
        if self.input_device is None or self.output_device is None:
            raise RuntimeError("Input/output device is not selected.")

        in_dev = sd.query_devices(self.input_device)
        out_dev = sd.query_devices(self.output_device)
        max_out_channels = int(out_dev["max_output_channels"])
        out_channel_candidates = [1]
        if max_out_channels >= 2:
            out_channel_candidates.append(2)

        # 장치 기본 샘플레이트를 우선 시도하고, 대표 레이트를 순차적으로 fallback
        candidates = [
            int(round(in_dev["default_samplerate"])),
            int(round(out_dev["default_samplerate"])),
            48000,
            44100,
            32000,
        ]
        checked = []
        for rate in candidates:
            if rate in checked:
                continue
            checked.append(rate)
            for out_channels in out_channel_candidates:
                try:
                    sd.check_input_settings(
                        device=self.input_device,
                        channels=self.input_channels,
                        samplerate=rate,
                    )
                    sd.check_output_settings(
                        device=self.output_device,
                        channels=out_channels,
                        samplerate=rate,
                    )
                    return rate, out_channels
                except Exception:
                    continue

        raise RuntimeError("No compatible stream configuration found for devices.")

    def start(self) -> int:
        """오디오 스트림을 시작하고 실제 동작 샘플레이트를 반환"""
        with self.lock:
            if self.stream is not None:
                return self.sample_rate
            if self.input_device is None or self.output_device is None:
                raise RuntimeError("Input/output device is not selected.")

        self.sample_rate, self.output_channels = self._find_compatible_stream_config()
        self._reset_effect_buffers()
        self.last_stream_status = ""

        # 입력/출력 장치를 동시에 여는 full-duplex 스트림.
        self.stream = sd.Stream(
            device=(self.input_device, self.output_device),
            channels=(self.input_channels, self.output_channels),
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            latency="high",
            dtype="float32",
            callback=self._audio_callback,
        )
        self.stream.start()
        return self.sample_rate

    def stop(self) -> None:
        """오디오 스트림을 안전하게 중지하고 녹음 상태를 정리"""
        with self.lock:
            stream = self.stream
            self.stream = None
            self.is_recording = False
            self.recorded_chunks = []
            self.active_record_output_mode = self.record_output_mode

        if stream is not None:
            stream.stop()
            stream.close()

    def is_running(self) -> bool:
        """스트림이 실제 활성 상태인지 확인"""
        with self.lock:
            stream = self.stream
        if stream is None:
            return False
        try:
            return bool(stream.active)
        except Exception:
            return False

    # -------- 오디오 처리 --------
    def _audio_callback(
        self,
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time_info,
        status: sd.CallbackFlags,
    ) -> None:
        """사운드카드 콜백: 입력 신호 처리, 출력 반영, 녹음 버퍼 저장을 수행"""
        with self.lock:
            params = copy.copy(self.params)
            recording_enabled = self.is_recording
            record_output_mode = self.active_record_output_mode
        if status:
            self.last_stream_status = str(status)

        if indata.size == 0:
            mono = np.zeros(frames, dtype=np.float32)
        else:
            mono = indata[:, 0].astype(np.float32, copy=True)

        processed = self._process_block(mono, params)
        # 녹음 중 무음 모드에서는 출력만 끄고, 녹음 데이터는 저장
        mute_output = recording_enabled and record_output_mode == "mute_while_recording"
        if mute_output:
            outdata.fill(0.0)
        else:
            outdata[:, 0] = processed
            if outdata.shape[1] > 1:
                outdata[:, 1:] = processed[:, np.newaxis]

        with self.lock:
            self.latest_output = processed
            if recording_enabled:
                self.recorded_chunks.append(processed.copy())

    def _process_block(self, samples: np.ndarray, params: EffectParams) -> np.ndarray:
        """한 블록의 오디오 샘플에 이펙트 체인을 순서대로 적용"""
        x = samples.astype(np.float32, copy=True)
        if params.input_gain != 1.0:
            x *= params.input_gain

        if params.distortion_drive > 1.001:
            x = np.tanh(x * params.distortion_drive).astype(np.float32)

        if params.reverb_wet > 0.001:
            x = self._apply_reverb(x, params.reverb_wet)

        if params.delay_wet > 0.001:
            x = self._apply_delay(
                x, params.delay_wet, params.delay_time_ms, params.delay_feedback
            )

        if params.echo_wet > 0.001:
            x = self._apply_echo(
                x, params.echo_wet, params.echo_time_ms, params.echo_feedback
            )

        if params.output_gain != 1.0:
            x *= params.output_gain

        # 최종 출력은 [-1.0, 1.0] 범위로 제한해 클리핑 왜곡을 방지한다.
        np.clip(x, -1.0, 1.0, out=x)
        return x

    def _apply_reverb(self, samples: np.ndarray, wet: float) -> np.ndarray:
        """짧은 멀티탭 버퍼를 이용해 간단한 리버브 효과를 적용"""
        wet = float(np.clip(wet, 0.0, 1.0))
        output = np.empty_like(samples)
        feedback = 0.35

        for i, dry in enumerate(samples):
            reverb_sum = 0.0
            for tap, gain in zip(self.reverb_taps, self.reverb_tap_gains):
                read_idx = (self.reverb_idx - tap) % self.max_reverb_samples
                reverb_sum += self.reverb_buffer[read_idx] * gain

            self.reverb_buffer[self.reverb_idx] = dry + (reverb_sum * feedback)
            self.reverb_idx = (self.reverb_idx + 1) % self.max_reverb_samples
            output[i] = (dry * (1.0 - wet)) + (reverb_sum * wet)

        return output

    def _apply_delay(
        self, samples: np.ndarray, wet: float, delay_ms: float, feedback: float
    ) -> np.ndarray:
        """딜레이 시간/피드백 파라미터 기반의 딜레이 효과를 적용"""
        wet = float(np.clip(wet, 0.0, 1.0))
        feedback = float(np.clip(feedback, 0.0, 0.95))
        delay_samples = int(delay_ms * self.sample_rate / 1000.0)
        delay_samples = max(1, min(delay_samples, self.max_delay_samples - 1))
        output = np.empty_like(samples)

        for i, dry in enumerate(samples):
            read_idx = (self.delay_idx - delay_samples) % self.max_delay_samples
            delayed = self.delay_buffer[read_idx]
            output[i] = (dry * (1.0 - wet)) + (delayed * wet)
            self.delay_buffer[self.delay_idx] = dry + (delayed * feedback)
            self.delay_idx = (self.delay_idx + 1) % self.max_delay_samples

        return output

    def _apply_echo(
        self, samples: np.ndarray, wet: float, echo_ms: float, feedback: float
    ) -> np.ndarray:
        """긴 지연 기반의 에코 효과를 적용"""
        wet = float(np.clip(wet, 0.0, 1.0))
        feedback = float(np.clip(feedback, 0.0, 0.95))
        echo_samples = int(echo_ms * self.sample_rate / 1000.0)
        echo_samples = max(1, min(echo_samples, self.max_echo_samples - 1))
        output = np.empty_like(samples)

        for i, dry in enumerate(samples):
            read_idx = (self.echo_idx - echo_samples) % self.max_echo_samples
            echoed = self.echo_buffer[read_idx]
            output[i] = (dry * (1.0 - wet)) + (echoed * wet)
            self.echo_buffer[self.echo_idx] = dry + (echoed * feedback)
            self.echo_idx = (self.echo_idx + 1) % self.max_echo_samples

        return output

    # -------- 녹음 데이터 접근 --------
    def get_latest_output(self) -> np.ndarray:
        """최근 출력 샘플 블록을 파형 표시용으로 복사 반환"""
        with self.lock:
            return self.latest_output.copy()

    def start_recording(self, record_output_mode: Optional[str] = None) -> None:
        """녹음 버퍼를 초기화하고 녹음 상태를 시작"""
        with self.lock:
            if record_output_mode in {"always", "mute_while_recording"}:
                self.record_output_mode = record_output_mode
            # 녹음 시작 시점의 모드를 현재 녹음 세션에 고정한다.
            self.active_record_output_mode = self.record_output_mode
            self.recorded_chunks = []
            self.is_recording = True

    def stop_recording(self) -> Tuple[np.ndarray, int]:
        """녹음을 종료하고 누적된 오디오 데이터와 샘플레이트를 반환"""
        with self.lock:
            self.is_recording = False
            self.active_record_output_mode = self.record_output_mode
            chunks = self.recorded_chunks
            self.recorded_chunks = []
            rate = self.sample_rate

        if not chunks:
            return np.array([], dtype=np.float32), rate
        return np.concatenate(chunks).astype(np.float32), rate


# ============================================================
# GUI 계층 (PyQt 메인 윈도우)
# ============================================================
class MainWindow(QMainWindow):
    """오디오 엔진 제어, 사용자 입력, 상태 표시를 담당하는 메인 UI"""

    # UI 문구 다국어 테이블. _t()를 통해 현재 언어의 문자열을 나타냄
    TRANSLATIONS = {
        "ko": {
            "window_title": "온단비 VE 스튜디오",
            "group_audio_devices": "오디오 장치",
            "group_effects_gain": "이펙트 및 게인",
            "group_waveforms": "파형",
            "group_recording": "녹음",
            "tab_effects": "이펙트",
            "tab_studio": "스튜디오",
            "tab_waveforms": "파형 보기",
            "tab_recording": "녹음/재생",
            "label_input_mic": "입력 마이크",
            "label_output_speaker": "출력 스피커",
            "label_language": "언어",
            "label_theme": "테마",
            "label_process_end_mode": "처리 종료 모드",
            "theme_light": "라이트 모드",
            "theme_dark": "다크 모드",
            "device_unassigned": "장치 미할당",
            "btn_refresh_devices": "장치 새로고침",
            "btn_start_processing": "처리 시작",
            "btn_stop_processing": "처리 중지",
            "process_end_mode_auto": "처리종료 자동",
            "process_end_mode_manual": "처리종료 수동",
            "status_stopped": "중지됨",
            "status_running_sr": "{samplerate} Hz 실행 중",
            "status_running_flag": "실행 중 ({status})",
            "status_stopped_error": "오류로 중지됨 ({status})",
            "slider_input_gain": "입력 게인",
            "slider_output_gain": "출력 게인",
            "slider_distortion_drive": "디스토션 드라이브",
            "slider_reverb_mix": "리버브 믹스",
            "slider_delay_mix": "딜레이 믹스",
            "slider_delay_time": "딜레이 시간",
            "slider_delay_feedback": "딜레이 피드백",
            "slider_echo_mix": "에코 믹스",
            "slider_echo_time": "에코 시간",
            "slider_echo_feedback": "에코 피드백",
            "plot_live_title": "실시간 처리 신호",
            "plot_recorded_title": "최근 녹음 신호",
            "plot_time_label": "시간",
            "label_save_path": "저장 파일",
            "label_record_output_mode": "모니터링 모드",
            "label_playback_position": "재생 위치",
            "btn_browse_file": "파일 선택",
            "btn_pick_folder": "폴더 선택",
            "record_output_mode_always": "실시간 출력",
            "record_output_mode_mute_while_recording": "녹음 중 무음 (재생 시 출력)",
            "btn_start_recording": "녹음 시작",
            "btn_stop_and_save": "중지 후 저장",
            "btn_play_pause_play": "재생",
            "btn_play_pause_pause": "일시정지",
            "btn_stop_playback": "정지",
            "status_no_recording": "아직 녹음된 파일이 없습니다.",
            "status_recording": "녹음 중...",
            "status_no_audio_captured": "녹음을 중지했지만 캡처된 오디오가 없습니다.",
            "status_saved": "저장 완료: {path} ({duration:.2f}초)",
            "status_playing": "재생 중: {path}",
            "status_playback_paused": "일시정지: {path}",
            "status_playback_stopped": "정지(0초): {path}",
            "hint_virtual_routing": "팁: Discord/음성 녹음기 등 외부 앱은 가상 오디오 케이블 장치로 라우팅하세요.",
            "dialog_device_error": "장치 오류",
            "dialog_missing_device": "장치 선택 필요",
            "msg_select_input_output": "입력과 출력을 모두 선택하세요.",
            "dialog_audio_start_error": "오디오 시작 오류",
            "dialog_audio_stop_warning": "오디오 중지 경고",
            "dialog_save_as": "녹음 파일로 저장",
            "dialog_select_folder": "저장 폴더 선택",
            "filter_wav": "WAV 파일 (*.wav)",
            "filter_audio_save": "오디오 파일 (*.wav *.m4a);;WAV 파일 (*.wav);;M4A 파일 (*.m4a)",
            "dialog_stream_not_running": "스트림 미실행",
            "msg_start_processing_first": "먼저 오디오 처리를 시작한 뒤 녹음을 시작하세요.",
            "dialog_save_error": "저장 오류",
            "dialog_no_recording": "녹음 파일 없음",
            "msg_record_first": "먼저 녹음 후 저장을 진행하세요.",
            "dialog_playback_error": "재생 오류",
            "msg_select_output_before_playback": "재생 전에 출력 장치를 선택하세요.",
            "dialog_processing_active": "처리 실행 중",
            "msg_playback_while_processing_locked": "처리가 실행 중일 때는 자동으로 중지되지 않습니다. 재생하려면 먼저 처리 중지를 눌러주세요.",
            "dialog_stream_error": "오디오 스트림 오류",
            "status_playback_done": "재생 완료: {path}",
            "status_playback_position": "{current:.2f}초 / {total:.2f}초",
            "msg_m4a_dep_missing": "M4A 처리를 위해 `imageio-ffmpeg` 패키지가 필요합니다.",
            "msg_m4a_convert_failed": "M4A 변환/로딩에 실패했습니다: {reason}",
            "group_presets_settings": "프리셋 및 설정",
            "label_preset": "기본 프리셋",
            "btn_apply_preset": "프리셋 적용",
            "btn_save_settings": "세팅 저장",
            "btn_load_settings": "세팅 불러오기",
            "btn_reset_defaults": "기본값 초기화",
            "preset_custom": "사용자 직접 설정",
            "preset_stage": "무대공연",
            "preset_karaoke": "노래방",
            "preset_clean_boost": "클린 부스트",
            "dialog_save_settings": "세팅 저장",
            "dialog_load_settings": "세팅 불러오기",
            "filter_json": "JSON 파일 (*.json)",
            "dialog_settings_error": "세팅 오류",
            "status_preset_applied": "프리셋 적용 완료: {name}",
            "status_settings_saved": "세팅 저장 완료: {path}",
            "status_settings_loaded": "세팅 불러오기 완료: {path}",
            "status_settings_reset": "기본값으로 초기화했습니다.",
        },
        "en": {
            "window_title": "Ondanbi VE Studio",
            "group_audio_devices": "Audio Devices",
            "group_effects_gain": "Effects and Gain",
            "group_waveforms": "Waveforms",
            "group_recording": "Recording",
            "tab_effects": "Effects",
            "tab_studio": "Studio",
            "tab_waveforms": "Waveforms",
            "tab_recording": "Record/Playback",
            "label_input_mic": "Input Mic",
            "label_output_speaker": "Output Speaker",
            "label_language": "Language",
            "label_theme": "Theme",
            "label_process_end_mode": "Process End Mode",
            "theme_light": "Light Mode",
            "theme_dark": "Dark Mode",
            "device_unassigned": "Unassigned",
            "btn_refresh_devices": "Refresh Devices",
            "btn_start_processing": "Start Processing",
            "btn_stop_processing": "Stop Processing",
            "process_end_mode_auto": "Auto Stop Processing",
            "process_end_mode_manual": "Manual Stop Processing",
            "status_stopped": "Stopped",
            "status_running_sr": "Running @ {samplerate} Hz",
            "status_running_flag": "Running ({status})",
            "status_stopped_error": "Stopped by error ({status})",
            "slider_input_gain": "Input Gain",
            "slider_output_gain": "Output Gain",
            "slider_distortion_drive": "Distortion Drive",
            "slider_reverb_mix": "Reverb Mix",
            "slider_delay_mix": "Delay Mix",
            "slider_delay_time": "Delay Time",
            "slider_delay_feedback": "Delay Feedback",
            "slider_echo_mix": "Echo Mix",
            "slider_echo_time": "Echo Time",
            "slider_echo_feedback": "Echo Feedback",
            "plot_live_title": "Live Processed Signal",
            "plot_recorded_title": "Last Recorded Signal",
            "plot_time_label": "Time",
            "label_save_path": "Save File",
            "label_record_output_mode": "Monitor Mode",
            "label_playback_position": "Playback Position",
            "btn_browse_file": "Browse File",
            "btn_pick_folder": "Choose Folder",
            "record_output_mode_always": "Live Monitor",
            "record_output_mode_mute_while_recording": "Mute While Recording (playback only)",
            "btn_start_recording": "Start Recording",
            "btn_stop_and_save": "Stop and Save",
            "btn_play_pause_play": "Play",
            "btn_play_pause_pause": "Pause",
            "btn_stop_playback": "Stop",
            "status_no_recording": "No recording yet.",
            "status_recording": "Recording...",
            "status_no_audio_captured": "Recording stopped (no audio captured).",
            "status_saved": "Saved: {path} ({duration:.2f} sec)",
            "status_playing": "Playing: {path}",
            "status_playback_paused": "Paused: {path}",
            "status_playback_stopped": "Stopped (0s): {path}",
            "hint_virtual_routing": "Tip: For external apps (Discord/Voice Recorder), route output to a virtual audio cable device.",
            "dialog_device_error": "Device Error",
            "dialog_missing_device": "Missing Device",
            "msg_select_input_output": "Select both input and output.",
            "dialog_audio_start_error": "Audio Start Error",
            "dialog_audio_stop_warning": "Audio Stop Warning",
            "dialog_save_as": "Save Recording As",
            "dialog_select_folder": "Select Save Folder",
            "filter_wav": "WAV files (*.wav)",
            "filter_audio_save": "Audio files (*.wav *.m4a);;WAV files (*.wav);;M4A files (*.m4a)",
            "dialog_stream_not_running": "Stream Not Running",
            "msg_start_processing_first": "Start audio processing first, then record processed sound.",
            "dialog_save_error": "Save Error",
            "dialog_no_recording": "No Recording",
            "msg_record_first": "Record and save audio first.",
            "dialog_playback_error": "Playback Error",
            "msg_select_output_before_playback": "Select an output device before playback.",
            "dialog_processing_active": "Processing Active",
            "msg_playback_while_processing_locked": "Processing will not auto-stop while running. Click Stop Processing first, then play recording.",
            "dialog_stream_error": "Audio Stream Error",
            "status_playback_done": "Playback finished: {path}",
            "status_playback_position": "{current:.2f}s / {total:.2f}s",
            "msg_m4a_dep_missing": "`imageio-ffmpeg` is required for M4A support.",
            "msg_m4a_convert_failed": "M4A conversion/loading failed: {reason}",
            "group_presets_settings": "Presets and Settings",
            "label_preset": "Built-in Presets",
            "btn_apply_preset": "Apply Preset",
            "btn_save_settings": "Save Settings",
            "btn_load_settings": "Load Settings",
            "btn_reset_defaults": "Reset Defaults",
            "preset_custom": "Custom",
            "preset_stage": "Stage Performance",
            "preset_karaoke": "Karaoke",
            "preset_clean_boost": "Clean Boost",
            "dialog_save_settings": "Save Settings",
            "dialog_load_settings": "Load Settings",
            "filter_json": "JSON files (*.json)",
            "dialog_settings_error": "Settings Error",
            "status_preset_applied": "Preset applied: {name}",
            "status_settings_saved": "Settings saved: {path}",
            "status_settings_loaded": "Settings loaded: {path}",
            "status_settings_reset": "Reset to defaults.",
        },
    }

    PRESET_VALUES: Dict[str, Dict[str, int]] = {
        "stage": {
            "input_gain": 120,
            "output_gain": 120,
            "distortion": 35,
            "reverb": 35,
            "delay_mix": 15,
            "delay_time": 260,
            "delay_feedback": 25,
            "echo_mix": 10,
            "echo_time": 420,
            "echo_feedback": 20,
        },
        "karaoke": {
            "input_gain": 110,
            "output_gain": 115,
            "distortion": 0,
            "reverb": 24,
            "delay_mix": 12,
            "delay_time": 180,
            "delay_feedback": 18,
            "echo_mix": 8,
            "echo_time": 320,
            "echo_feedback": 12,
        },
        "clean_boost": {
            "input_gain": 125,
            "output_gain": 125,
            "distortion": 0,
            "reverb": 0,
            "delay_mix": 0,
            "delay_time": 0,
            "delay_feedback": 0,
            "echo_mix": 0,
            "echo_time": 0,
            "echo_feedback": 0,
        },
    }

    def __init__(self) -> None:
        """엔진/상태를 초기화하고 UI를 구성한 뒤 타이머 갱신을 시작"""
        super().__init__()
        self.resize(1280, 940)
        # 실시간 파형 렌더 부하를 줄여 오디오 콜백 안정성 확보
        pg.setConfigOptions(antialias=False)
        pg.setConfigOption("background", "#fbfdff")
        pg.setConfigOption("foreground", "#344861")

        self.engine = AudioEngine()
        self.last_recording_path: Optional[Path] = None
        self.current_language = "ko"
        self.current_theme = "light"
        self.slider_title_labels: dict[str, QLabel] = {}
        self.preset_keys_in_order = ["custom", "stage", "karaoke", "clean_boost"]
        self.process_end_mode = "auto"
        self.playback_stream: Optional[sd.OutputStream] = None
        self.playback_audio: Optional[np.ndarray] = None
        self.playback_samplerate = 0
        self.playback_total_frames = 0
        self.playback_current_frame = 0
        self.playback_source_path: Optional[Path] = None
        self.playback_output_device: Optional[int] = None
        self.playback_lock = threading.Lock()
        self.playback_cursor_internal_update = False
        self.live_wave_visible = False
        self.last_cursor_frame_synced = -1
        self.stream_button_running_state: Optional[bool] = None
        self.playback_button_signature: Optional[Tuple[bool, bool, bool, str]] = None
        self.stream_expected_running = False

        self._build_ui()
        self._apply_theme()
        self._apply_language()
        self._load_devices()
        self._sync_all_params()

        self.wave_timer = QTimer(self)
        # 파형 갱신 빈도를 약간 낮춰 UI 부하와 GIL 경합을 줄인다.
        self.wave_timer.setInterval(80)
        self.wave_timer.timeout.connect(self._update_live_waveform)
        self.wave_timer.start()

    # -------- 언어/모드 상태 처리 --------
    def _t(self, key: str, **kwargs) -> str:
        """현재 선택 언어의 번역 문자열을 가져오고 포맷팅"""
        lang_table = self.TRANSLATIONS.get(
            self.current_language, self.TRANSLATIONS["en"]
        )
        text = lang_table.get(key, key)
        if kwargs:
            return text.format(**kwargs)
        return text

    def _on_language_changed(self) -> None:
        """언어 콤보 박스 변경 이벤트를 처리해 UI 문구를 갱신"""
        lang = self.language_combo.currentData()
        if isinstance(lang, str):
            self.current_language = lang
            self._apply_language()

    def _toggle_theme(self) -> None:
        """달/태양 버튼 클릭 시 다크/라이트 테마를 전환."""
        self.current_theme = "dark" if self.current_theme == "light" else "light"
        self._apply_theme()

    @staticmethod
    def _create_theme_icon(kind: str) -> QIcon:
        """테마 토글용 달/태양 아이콘 생성."""
        pixmap = QPixmap(20, 20)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        if kind == "moon":
            moon_color = QColor("#334155")
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(moon_color)
            painter.drawEllipse(3, 3, 14, 14)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.drawEllipse(8, 2, 12, 16)
        else:
            sun_color = QColor("#f59e0b")
            pen = QPen(sun_color)
            pen.setWidth(2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.setBrush(sun_color)
            painter.drawEllipse(6, 6, 8, 8)
            rays = [
                ((10, 1), (10, 4)),
                ((10, 16), (10, 19)),
                ((1, 10), (4, 10)),
                ((16, 10), (19, 10)),
                ((3, 3), (5, 5)),
                ((15, 15), (17, 17)),
                ((15, 5), (17, 3)),
                ((3, 17), (5, 15)),
            ]
            for start, end in rays:
                painter.drawLine(start[0], start[1], end[0], end[1])

        painter.end()
        return QIcon(pixmap)

    def _refresh_theme_toggle_button(self) -> None:
        """현재 테마 상태에 맞춰 토글 버튼의 아이콘/툴팁을 갱신."""
        if not hasattr(self, "theme_toggle_btn"):
            return
        if self.current_theme == "light":
            self.theme_toggle_btn.setIcon(self._create_theme_icon("moon"))
            self.theme_toggle_btn.setToolTip(self._t("theme_dark"))
            self.theme_toggle_btn.setStatusTip(self._t("theme_dark"))
        else:
            self.theme_toggle_btn.setIcon(self._create_theme_icon("sun"))
            self.theme_toggle_btn.setToolTip(self._t("theme_light"))
            self.theme_toggle_btn.setStatusTip(self._t("theme_light"))

    def _toggle_stream(self) -> None:
        """처리 시작/중지 단일 버튼 토글 동작."""
        if self.engine.is_running():
            self._stop_stream()
        else:
            self._start_stream()

    def _on_record_output_mode_changed(self) -> None:
        """모니터링 모드 콤보 박스 변경을 엔진 설정으로 반영."""
        self.engine.set_record_output_mode(self._get_selected_record_output_mode())

    def _on_process_end_mode_changed(self) -> None:
        """처리 종료 모드 콤보 박스 변경을 내부 상태에 반영."""
        mode = self.process_end_mode_combo.currentData()
        if mode in {"auto", "manual"}:
            self.process_end_mode = str(mode)

    def _get_selected_record_output_mode(self) -> str:
        """UI 콤보에서 현재 선택된 녹음 출력 모드를 반환."""
        mode = self.record_mode_combo.currentData()
        if mode in {"always", "mute_while_recording"}:
            return str(mode)
        return "mute_while_recording"

    @staticmethod
    def _settings_default_path() -> Path:
        """세팅 JSON의 기본 저장 경로를 반환."""
        return Path.cwd() / "voice_settings.json"

    def _rebuild_preset_combo(self) -> None:
        """현재 언어 기준으로 프리셋 콤보 항목을 다시 구성."""
        previous = (
            self.preset_combo.currentData() if hasattr(self, "preset_combo") else None
        )
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        for preset_key in self.preset_keys_in_order:
            self.preset_combo.addItem(self._t(f"preset_{preset_key}"), preset_key)
        restored_index = self.preset_combo.findData(previous)
        if restored_index < 0:
            restored_index = self.preset_combo.findData("custom")
        if restored_index >= 0:
            self.preset_combo.setCurrentIndex(restored_index)
        self.preset_combo.blockSignals(False)

    def _capture_slider_settings(self) -> Dict[str, int]:
        """현재 슬라이더 값을 직렬화 가능한 dict로 수집."""
        return {
            "input_gain": int(self.input_gain_slider.value()),
            "output_gain": int(self.output_gain_slider.value()),
            "distortion": int(self.distortion_slider.value()),
            "reverb": int(self.reverb_slider.value()),
            "delay_mix": int(self.delay_mix_slider.value()),
            "delay_time": int(self.delay_time_slider.value()),
            "delay_feedback": int(self.delay_feedback_slider.value()),
            "echo_mix": int(self.echo_mix_slider.value()),
            "echo_time": int(self.echo_time_slider.value()),
            "echo_feedback": int(self.echo_feedback_slider.value()),
        }

    def _clamp_slider_value(self, slider: QSlider, value: Any) -> int:
        """슬라이더 범위에 맞게 값을 정수로 보정."""
        try:
            as_int = int(round(float(value)))
        except Exception:
            as_int = slider.value()
        return int(np.clip(as_int, slider.minimum(), slider.maximum()))

    def _apply_slider_settings(self, values: Dict[str, Any]) -> None:
        """슬라이더 값 dict를 UI와 엔진 파라미터에 반영."""
        mapping: Dict[str, QSlider] = {
            "input_gain": self.input_gain_slider,
            "output_gain": self.output_gain_slider,
            "distortion": self.distortion_slider,
            "reverb": self.reverb_slider,
            "delay_mix": self.delay_mix_slider,
            "delay_time": self.delay_time_slider,
            "delay_feedback": self.delay_feedback_slider,
            "echo_mix": self.echo_mix_slider,
            "echo_time": self.echo_time_slider,
            "echo_feedback": self.echo_feedback_slider,
        }
        for key, slider in mapping.items():
            if key in values:
                slider.setValue(self._clamp_slider_value(slider, values[key]))

    def _collect_settings_payload(self) -> Dict[str, Any]:
        """현재 UI 상태를 JSON 저장용 payload로 구성."""
        return {
            "schema_version": 1,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "language": str(self.language_combo.currentData() or self.current_language),
            "theme": self.current_theme,
            "process_end_mode": str(
                self.process_end_mode_combo.currentData() or "auto"
            ),
            "record_output_mode": self._get_selected_record_output_mode(),
            "record_save_path": self.path_edit.text().strip(),
            "preset_key": str(self.preset_combo.currentData() or "custom"),
            "sliders": self._capture_slider_settings(),
        }

    def _apply_settings_payload(self, payload: Dict[str, Any]) -> None:
        """JSON payload를 UI 상태로 복원."""
        if not isinstance(payload, dict):
            raise RuntimeError("Invalid settings format.")

        sliders = payload.get("sliders", {})
        if isinstance(sliders, dict):
            self._apply_slider_settings(sliders)

        language = payload.get("language")
        if isinstance(language, str):
            idx = self.language_combo.findData(language)
            if idx >= 0:
                self.language_combo.setCurrentIndex(idx)

        theme = payload.get("theme")
        if isinstance(theme, str) and theme in {"light", "dark"}:
            self.current_theme = theme
            self._apply_theme()

        process_mode = payload.get("process_end_mode")
        if isinstance(process_mode, str):
            idx = self.process_end_mode_combo.findData(process_mode)
            if idx >= 0:
                self.process_end_mode_combo.setCurrentIndex(idx)

        record_mode = payload.get("record_output_mode")
        if isinstance(record_mode, str):
            idx = self.record_mode_combo.findData(record_mode)
            if idx >= 0:
                self.record_mode_combo.setCurrentIndex(idx)
                self.engine.set_record_output_mode(record_mode)

        save_path = payload.get("record_save_path")
        if isinstance(save_path, str) and save_path.strip():
            self.path_edit.setText(save_path.strip())

        preset_key = payload.get("preset_key")
        if isinstance(preset_key, str):
            idx = self.preset_combo.findData(preset_key)
            if idx >= 0:
                self.preset_combo.setCurrentIndex(idx)

        self._sync_all_params()

    def _save_settings_to_file(self) -> None:
        """현재 세팅을 JSON 파일로 저장."""
        default_path = str(self._settings_default_path())
        target, _ = QFileDialog.getSaveFileName(
            self,
            self._t("dialog_save_settings"),
            default_path,
            self._t("filter_json"),
        )
        if not target:
            return
        save_path = Path(target).expanduser()
        if save_path.suffix.lower() != ".json":
            save_path = save_path.with_suffix(".json")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        payload = self._collect_settings_payload()
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            QMessageBox.critical(self, self._t("dialog_settings_error"), str(exc))
            return

        self.record_status.setText(self._t("status_settings_saved", path=save_path))

    def _load_settings_from_file(self) -> None:
        """JSON 세팅 파일을 로드해 현재 UI에 적용."""
        default_path = str(self._settings_default_path())
        target, _ = QFileDialog.getOpenFileName(
            self,
            self._t("dialog_load_settings"),
            default_path,
            self._t("filter_json"),
        )
        if not target:
            return
        load_path = Path(target).expanduser()
        try:
            with open(load_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self._apply_settings_payload(payload)
        except Exception as exc:
            QMessageBox.critical(self, self._t("dialog_settings_error"), str(exc))
            return

        self.record_status.setText(self._t("status_settings_loaded", path=load_path))

    def _apply_selected_preset(self) -> None:
        """선택한 내장 프리셋을 현재 세팅에 반영."""
        preset_key = self.preset_combo.currentData()
        if not isinstance(preset_key, str) or preset_key == "custom":
            return
        values = self.PRESET_VALUES.get(preset_key)
        if values is None:
            return
        self._apply_slider_settings(values)
        self._sync_all_params()
        self.record_status.setText(
            self._t("status_preset_applied", name=self.preset_combo.currentText())
        )

    def _reset_to_defaults(self) -> None:
        """입출력 게인 제외 전체 0 기본값으로 복원."""
        default_values = {
            "input_gain": 100,
            "output_gain": 100,
            "distortion": 0,
            "reverb": 0,
            "delay_mix": 0,
            "delay_time": 0,
            "delay_feedback": 0,
            "echo_mix": 0,
            "echo_time": 0,
            "echo_feedback": 0,
        }
        self._apply_slider_settings(default_values)
        custom_index = self.preset_combo.findData("custom")
        if custom_index >= 0:
            self.preset_combo.setCurrentIndex(custom_index)
        self._sync_all_params()
        self.record_status.setText(self._t("status_settings_reset"))

    def _set_icon_button(
        self,
        button: QPushButton,
        icon_type: QStyle.StandardPixmap,
        text: str,
    ) -> None:
        """아이콘 + 텍스트 버튼 공통 스타일."""
        button.setIcon(self.style().standardIcon(icon_type))
        button.setText(text)
        button.setToolTip(text)
        button.setStatusTip(text)
        button.setMinimumHeight(32)

    @staticmethod
    def _create_shape_icon(shape: str, color: str) -> QIcon:
        """단색 도형 아이콘(원/사각형)을 생성."""
        pixmap = QPixmap(18, 18)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(color))
        if shape == "circle":
            painter.drawEllipse(2, 2, 14, 14)
        else:
            painter.drawRect(3, 3, 12, 12)
        painter.end()
        return QIcon(pixmap)

    def _set_record_button_sizes(self) -> None:
        """녹음/재생 컨트롤 버튼 크기를 동일하게 고정."""
        for button in (
            self.rec_start_btn,
            self.rec_stop_btn,
            self.play_btn,
            self.stop_playback_btn,
        ):
            button.setMinimumWidth(116)
            button.setMaximumWidth(116)
            button.setMinimumHeight(34)

    def _refresh_stream_toggle_button(self, force: bool = False) -> None:
        """현재 처리 상태에 따라 시작 버튼의 아이콘/문구를 토글."""
        is_running = self.engine.is_running()
        if (not force) and self.stream_button_running_state == is_running:
            return
        self.stream_button_running_state = is_running

        if is_running:
            self.start_btn.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop)
            )
            self.start_btn.setText(self._t("btn_stop_processing"))
            self.start_btn.setToolTip(self._t("btn_stop_processing"))
            self.start_btn.setStatusTip(self._t("btn_stop_processing"))
        else:
            self.start_btn.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
            )
            self.start_btn.setText(self._t("btn_start_processing"))
            self.start_btn.setToolTip(self._t("btn_start_processing"))
            self.start_btn.setStatusTip(self._t("btn_start_processing"))

        # 상단 주요 버튼(장치 새로고침/처리 시작)의 크기를 동일하게 고정
        self.refresh_btn.setMinimumWidth(132)
        self.refresh_btn.setMaximumWidth(132)
        self.refresh_btn.setMinimumHeight(34)
        self.start_btn.setMinimumWidth(132)
        self.start_btn.setMaximumWidth(132)
        self.start_btn.setMinimumHeight(34)

    def _apply_theme(self) -> None:
        """현재 선택한 다크/라이트 테마를 앱 전체에 적용."""
        if self.current_theme == "dark":
            colors = {
                "window_bg": "#111827",
                "text": "#e5e7eb",
                "card_bg": "#1f2937",
                "card_border": "#374151",
                "title": "#93c5fd",
                "accent": "#60a5fa",
                "accent_hover": "#3b82f6",
                "accent_pressed": "#2563eb",
                "disabled_bg": "#4b5563",
                "disabled_fg": "#9ca3af",
                "input_bg": "#111827",
                "input_border": "#4b5563",
                "combo_popup_bg": "#1f2937",
                "combo_popup_sel": "#2563eb",
                "slider_track": "#334155",
                "slider_fill": "#60a5fa",
                "slider_handle_border": "#93c5fd",
                "tab_bg": "#243244",
                "tab_text": "#c8d7ea",
                "tab_hover": "#334155",
                "status_bg": "#0f2740",
                "status_fg": "#93c5fd",
                "status_border": "#375a7f",
                "plot_bg": "#111827",
                "plot_fg": "#cbd5e1",
                "live_color": "#34d399",
                "recorded_color": "#60a5fa",
                "cursor_color": "#f59e0b",
            }
        else:
            colors = {
                "window_bg": "#f3f6fb",
                "text": "#1f2a37",
                "card_bg": "#ffffff",
                "card_border": "#d4deea",
                "title": "#27435f",
                "accent": "#2a7bd5",
                "accent_hover": "#1f6dbe",
                "accent_pressed": "#185796",
                "disabled_bg": "#bcc8d8",
                "disabled_fg": "#eef3f9",
                "input_bg": "#fbfdff",
                "input_border": "#c7d3e2",
                "combo_popup_bg": "#ffffff",
                "combo_popup_sel": "#2a7bd5",
                "slider_track": "#d8e1ed",
                "slider_fill": "#2a7bd5",
                "slider_handle_border": "#165491",
                "tab_bg": "#e7eef7",
                "tab_text": "#35506b",
                "tab_hover": "#dfe9f6",
                "status_bg": "#edf4ff",
                "status_fg": "#1e4f80",
                "status_border": "#c8daf1",
                "plot_bg": "#fbfdff",
                "plot_fg": "#344861",
                "live_color": "#3f9f5f",
                "recorded_color": "#2f5fa0",
                "cursor_color": "#ff8a00",
            }

        self.setStyleSheet(
            f"""
            QMainWindow {{
                background: {colors["window_bg"]};
            }}
            QWidget {{
                color: {colors["text"]};
                font-family: "Segoe UI", "Noto Sans KR";
                font-size: 10pt;
            }}
            QGroupBox {{
                background: {colors["card_bg"]};
                border: 1px solid {colors["card_border"]};
                border-radius: 12px;
                margin-top: 12px;
                font-weight: 600;
                padding: 8px 10px 10px 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
                color: {colors["title"]};
            }}
            QPushButton {{
                background: {colors["accent"]};
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 6px 12px;
                min-height: 28px;
            }}
            QPushButton:hover {{
                background: {colors["accent_hover"]};
            }}
            QPushButton:pressed {{
                background: {colors["accent_pressed"]};
            }}
            QPushButton:disabled {{
                background: {colors["disabled_bg"]};
                color: {colors["disabled_fg"]};
            }}
            QPushButton#themeToggle {{
                background: {colors["input_bg"]};
                border: 1px solid {colors["input_border"]};
                border-radius: 8px;
                min-width: 36px;
                max-width: 36px;
                min-height: 36px;
                max-height: 36px;
                padding: 0px;
            }}
            QPushButton#themeToggle:hover {{
                background: {colors["tab_hover"]};
            }}
            QPushButton#themeToggle:pressed {{
                background: {colors["card_border"]};
            }}
            QLineEdit, QComboBox {{
                background: {colors["input_bg"]};
                color: {colors["text"]};
                border: 1px solid {colors["input_border"]};
                border-radius: 8px;
                padding: 4px 8px;
                min-height: 28px;
                selection-background-color: {colors["accent"]};
                selection-color: #ffffff;
            }}
            QComboBox::drop-down {{
                width: 24px;
                border: 0px;
                border-left: 1px solid {colors["input_border"]};
                background: {colors["card_border"]};
                border-top-right-radius: 8px;
                border-bottom-right-radius: 8px;
            }}
            QComboBox QAbstractItemView {{
                background: {colors["combo_popup_bg"]};
                color: {colors["text"]};
                border: 1px solid {colors["input_border"]};
                selection-background-color: {colors["combo_popup_sel"]};
                selection-color: #ffffff;
                outline: 0;
            }}
            QSlider::groove:horizontal {{
                border: none;
                height: 10px;
                background: {colors["slider_track"]};
                border-radius: 5px;
            }}
            QSlider::sub-page:horizontal {{
                height: 10px;
                background: {colors["slider_fill"]};
                border-radius: 5px;
            }}
            QSlider::add-page:horizontal {{
                height: 10px;
                background: {colors["slider_track"]};
                border-radius: 5px;
            }}
            QSlider::handle:horizontal {{
                background: {colors["slider_fill"]};
                border: 1px solid {colors["slider_handle_border"]};
                width: 18px;
                margin: -7px 0;
                border-radius: 9px;
            }}
            QTabWidget::pane {{
                border: 1px solid {colors["card_border"]};
                border-radius: 12px;
                background: {colors["card_bg"]};
                top: -1px;
            }}
            QTabBar::tab {{
                background: {colors["tab_bg"]};
                color: {colors["tab_text"]};
                border: 1px solid {colors["card_border"]};
                border-bottom: none;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                min-width: 130px;
                padding: 8px 14px;
                margin-right: 4px;
            }}
            QTabBar::tab:selected {{
                background: {colors["accent"]};
                color: #ffffff;
                border-color: {colors["accent"]};
            }}
            QTabBar::tab:!selected:hover {{
                background: {colors["tab_hover"]};
            }}
            QLabel#statusPill {{
                background: {colors["status_bg"]};
                color: {colors["status_fg"]};
                border: 1px solid {colors["status_border"]};
                border-radius: 8px;
                padding: 5px 10px;
                font-weight: 600;
            }}
            """
        )

        pg.setConfigOption("background", colors["plot_bg"])
        pg.setConfigOption("foreground", colors["plot_fg"])
        if hasattr(self, "live_plot") and hasattr(self, "recorded_plot"):
            self.live_plot.setBackground(colors["plot_bg"])
            self.recorded_plot.setBackground(colors["plot_bg"])
            axis_pen = pg.mkPen(colors["plot_fg"], width=1)
            for plot_widget in (self.live_plot, self.recorded_plot):
                for axis_name in ("bottom", "left"):
                    axis = plot_widget.getAxis(axis_name)
                    axis.setPen(axis_pen)
                    axis.setTextPen(axis_pen)
            if hasattr(self, "live_curve"):
                self.live_curve.setPen(pg.mkPen(colors["live_color"], width=1.8))
            if hasattr(self, "recorded_curve"):
                self.recorded_curve.setPen(
                    pg.mkPen(colors["recorded_color"], width=1.3)
                )
            if hasattr(self, "playback_cursor"):
                self.playback_cursor.setPen(pg.mkPen(colors["cursor_color"], width=2))
        self._refresh_theme_toggle_button()

    # -------- UI 구성 --------
    def _build_ui(self) -> None:
        """장치/이펙트/파형/녹음 UI를 생성하고 시그널을 연결"""
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setSpacing(12)
        root_layout.setContentsMargins(14, 14, 14, 14)

        # 1) 장치/언어/처리 시작-중지 영역
        self.device_group = QGroupBox()
        device_layout = QGridLayout(self.device_group)
        device_layout.setHorizontalSpacing(10)
        device_layout.setVerticalSpacing(8)
        self.input_label = QLabel()
        self.output_label = QLabel()
        self.language_label = QLabel()
        self.process_end_mode_label = QLabel()
        self.input_combo = QComboBox()
        self.output_combo = QComboBox()
        self.language_combo = QComboBox()
        self.language_combo.addItem("한국어", "ko")
        self.language_combo.addItem("English", "en")
        self.language_combo.currentIndexChanged.connect(self._on_language_changed)
        self.language_combo.setCurrentIndex(0)
        self.process_end_mode_combo = QComboBox()
        self.process_end_mode_combo.addItem("", "auto")
        self.process_end_mode_combo.addItem("", "manual")
        self.process_end_mode_combo.currentIndexChanged.connect(
            self._on_process_end_mode_changed
        )
        self.process_end_mode_combo.setCurrentIndex(0)
        self.process_end_mode = "auto"

        self.refresh_btn = QPushButton()
        self.refresh_btn.clicked.connect(self._load_devices)
        self.start_btn = QPushButton()
        self.start_btn.clicked.connect(self._toggle_stream)
        self.stop_btn = QPushButton()
        self.stop_btn.clicked.connect(self._stop_stream)
        self.stop_btn.setEnabled(False)
        self.stop_btn.hide()
        self.stream_label = QLabel()
        self.stream_label.setObjectName("statusPill")
        self.stream_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stream_label.setMinimumWidth(150)
        self.stream_label.setMaximumWidth(150)
        self.theme_toggle_btn = QPushButton()
        self.theme_toggle_btn.setObjectName("themeToggle")
        self.theme_toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.theme_toggle_btn.setFixedSize(36, 36)
        self.theme_toggle_btn.setIconSize(QSize(20, 20))
        self.theme_toggle_btn.clicked.connect(self._toggle_theme)

        device_layout.addWidget(self.input_label, 0, 0)
        device_layout.addWidget(self.input_combo, 0, 1)
        device_layout.addWidget(self.output_label, 1, 0)
        device_layout.addWidget(self.output_combo, 1, 1)
        device_layout.addWidget(self.language_label, 2, 0)
        device_layout.addWidget(self.language_combo, 2, 1)
        device_layout.addWidget(self.process_end_mode_label, 3, 0)
        device_layout.addWidget(self.process_end_mode_combo, 3, 1)
        device_layout.addWidget(self.refresh_btn, 0, 2)
        device_layout.addWidget(self.start_btn, 1, 2, 1, 2)
        device_layout.addWidget(self.stream_label, 0, 4, 2, 1)
        device_layout.addWidget(
            self.theme_toggle_btn,
            3,
            4,
            1,
            1,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
        )
        device_layout.setColumnStretch(1, 1)

        # 2) 프리셋/세팅 저장-불러오기 영역
        self.presets_group = QGroupBox()
        presets_layout = QGridLayout(self.presets_group)
        presets_layout.setHorizontalSpacing(10)
        presets_layout.setVerticalSpacing(8)
        self.preset_label = QLabel()
        self.preset_combo = QComboBox()
        self._rebuild_preset_combo()
        self.apply_preset_btn = QPushButton()
        self.apply_preset_btn.clicked.connect(self._apply_selected_preset)
        self.save_settings_btn = QPushButton()
        self.save_settings_btn.clicked.connect(self._save_settings_to_file)
        self.load_settings_btn = QPushButton()
        self.load_settings_btn.clicked.connect(self._load_settings_from_file)
        self.reset_defaults_btn = QPushButton()
        self.reset_defaults_btn.clicked.connect(self._reset_to_defaults)

        presets_layout.addWidget(self.preset_label, 0, 0)
        presets_layout.addWidget(self.preset_combo, 0, 1, 1, 2)
        presets_layout.addWidget(self.apply_preset_btn, 0, 3)
        presets_layout.addWidget(self.save_settings_btn, 0, 4)
        presets_layout.addWidget(self.load_settings_btn, 0, 5)
        presets_layout.addWidget(self.reset_defaults_btn, 0, 6)
        presets_layout.setColumnStretch(2, 1)

        top_row = QHBoxLayout()
        top_row.setSpacing(12)
        top_row.addWidget(self.device_group, 3)
        top_row.addWidget(self.presets_group, 2)
        root_layout.addLayout(top_row)

        self.main_tabs = QTabWidget()

        # 3) 이펙트 탭
        self.effects_tab = QWidget()
        effects_tab_layout = QVBoxLayout(self.effects_tab)
        effects_tab_layout.setContentsMargins(8, 8, 8, 8)
        effects_tab_layout.setSpacing(8)
        self.effects_group = QGroupBox()
        effects_layout = QGridLayout(self.effects_group)
        effects_layout.setHorizontalSpacing(12)
        effects_layout.setVerticalSpacing(8)
        row = 0
        self.input_gain_slider = self._add_slider(
            effects_layout,
            row,
            "slider_input_gain",
            0,
            300,
            100,
            lambda v: f"{v}%",
            lambda v: self.engine.set_param("input_gain", v / 100.0),
        )
        row += 1
        self.output_gain_slider = self._add_slider(
            effects_layout,
            row,
            "slider_output_gain",
            0,
            300,
            100,
            lambda v: f"{v}%",
            lambda v: self.engine.set_param("output_gain", v / 100.0),
        )
        row += 1
        self.distortion_slider = self._add_slider(
            effects_layout,
            row,
            "slider_distortion_drive",
            0,
            700,
            0,
            lambda v: f"{1.0 + (v / 100.0):.2f}x",
            lambda v: self.engine.set_param("distortion_drive", 1.0 + (v / 100.0)),
            text_to_value=self._parse_distortion_input_value,
        )
        row += 1
        self.reverb_slider = self._add_slider(
            effects_layout,
            row,
            "slider_reverb_mix",
            0,
            100,
            0,
            lambda v: f"{v}%",
            lambda v: self.engine.set_param("reverb_wet", v / 100.0),
        )
        row += 1
        self.delay_mix_slider = self._add_slider(
            effects_layout,
            row,
            "slider_delay_mix",
            0,
            100,
            0,
            lambda v: f"{v}%",
            lambda v: self.engine.set_param("delay_wet", v / 100.0),
        )
        row += 1
        self.delay_time_slider = self._add_slider(
            effects_layout,
            row,
            "slider_delay_time",
            0,
            1500,
            0,
            lambda v: f"{v} ms",
            lambda v: self.engine.set_param("delay_time_ms", float(v)),
        )
        row += 1
        self.delay_feedback_slider = self._add_slider(
            effects_layout,
            row,
            "slider_delay_feedback",
            0,
            95,
            0,
            lambda v: f"{v}%",
            lambda v: self.engine.set_param("delay_feedback", v / 100.0),
        )
        row += 1
        self.echo_mix_slider = self._add_slider(
            effects_layout,
            row,
            "slider_echo_mix",
            0,
            100,
            0,
            lambda v: f"{v}%",
            lambda v: self.engine.set_param("echo_wet", v / 100.0),
        )
        row += 1
        self.echo_time_slider = self._add_slider(
            effects_layout,
            row,
            "slider_echo_time",
            0,
            2500,
            0,
            lambda v: f"{v} ms",
            lambda v: self.engine.set_param("echo_time_ms", float(v)),
        )
        row += 1
        self.echo_feedback_slider = self._add_slider(
            effects_layout,
            row,
            "slider_echo_feedback",
            0,
            95,
            0,
            lambda v: f"{v}%",
            lambda v: self.engine.set_param("echo_feedback", v / 100.0),
        )
        effects_layout.setColumnStretch(1, 1)
        effects_tab_layout.addWidget(self.effects_group)
        self.main_tabs.addTab(self.effects_tab, "")

        # 4) 스튜디오 탭 (좌: 녹음/재생, 우: 파형 1:3)
        self.studio_tab = QWidget()
        studio_layout = QHBoxLayout(self.studio_tab)
        studio_layout.setContentsMargins(8, 8, 8, 8)
        studio_layout.setSpacing(10)

        self.rec_group = QGroupBox()
        rec_layout = QGridLayout(self.rec_group)
        rec_layout.setHorizontalSpacing(10)
        rec_layout.setVerticalSpacing(8)
        self.save_path_label = QLabel()
        self.path_edit = QLineEdit()
        self.path_edit.setText(str(Path.cwd() / "recording.wav"))
        self.browse_btn = QPushButton()
        self.browse_btn.clicked.connect(self._browse_record_file)
        self.pick_folder_btn = QPushButton()
        self.pick_folder_btn.clicked.connect(self._select_save_folder)
        self.record_mode_label = QLabel()
        self.record_mode_combo = QComboBox()
        self.record_mode_combo.addItem("", "always")
        self.record_mode_combo.addItem("", "mute_while_recording")
        self.record_mode_combo.currentIndexChanged.connect(
            self._on_record_output_mode_changed
        )
        self.record_mode_combo.setCurrentIndex(1)
        self.engine.set_record_output_mode(self._get_selected_record_output_mode())

        self.rec_start_btn = QPushButton()
        self.rec_start_btn.clicked.connect(self._start_recording)
        self.rec_stop_btn = QPushButton()
        self.rec_stop_btn.clicked.connect(self._stop_recording)
        self.rec_stop_btn.setEnabled(False)
        self.play_btn = QPushButton()
        self.play_btn.clicked.connect(self._play_recording)
        self.stop_playback_btn = QPushButton()
        self.stop_playback_btn.clicked.connect(self._stop_playback_to_start)
        self.stop_playback_btn.setEnabled(False)
        self._set_record_button_sizes()

        self.record_status = QLabel()
        self.record_status.setObjectName("statusPill")
        self.routing_hint = QLabel()

        rec_layout.addWidget(self.save_path_label, 0, 0)
        rec_layout.addWidget(self.path_edit, 0, 1, 1, 2)
        rec_layout.addWidget(self.browse_btn, 0, 3)
        rec_layout.addWidget(self.pick_folder_btn, 0, 4)
        rec_layout.addWidget(self.rec_start_btn, 1, 1)
        rec_layout.addWidget(self.rec_stop_btn, 1, 2)
        rec_layout.addWidget(self.play_btn, 1, 3)
        rec_layout.addWidget(self.stop_playback_btn, 1, 4)
        rec_layout.addWidget(self.record_mode_label, 2, 0)
        rec_layout.addWidget(self.record_mode_combo, 2, 1, 1, 3)
        rec_layout.addWidget(self.record_status, 3, 1, 1, 4)
        rec_layout.addWidget(self.routing_hint, 4, 1, 1, 4)
        rec_layout.setColumnStretch(1, 1)

        self.plot_group = QGroupBox()
        plot_layout = QVBoxLayout(self.plot_group)
        plot_layout.setSpacing(10)
        self.live_plot = pg.PlotWidget()
        self.live_plot.setYRange(-2.0, 2.0)
        self.live_plot.showGrid(x=True, y=True, alpha=0.22)
        self.live_plot.setMinimumHeight(160)
        self.live_plot.setMouseEnabled(x=False, y=False)
        self.live_plot.setMenuEnabled(False)
        live_item = self.live_plot.getPlotItem()
        live_item.hideButtons()
        live_item.setClipToView(True)
        live_item.setDownsampling(auto=True, mode="peak")
        self.live_curve = self.live_plot.plot(pen=pg.mkPen(color="#3f9f5f", width=1.5))

        self.recorded_plot = pg.PlotWidget()
        self.recorded_plot.setYRange(-2.0, 2.0)
        self.recorded_plot.showGrid(x=True, y=True, alpha=0.22)
        self.recorded_plot.setMinimumHeight(340)
        self.recorded_plot.setMouseEnabled(x=False, y=False)
        self.recorded_plot.setMenuEnabled(False)
        recorded_item = self.recorded_plot.getPlotItem()
        recorded_item.hideButtons()
        recorded_item.setClipToView(True)
        recorded_item.setDownsampling(auto=True, mode="peak")
        self.recorded_curve = self.recorded_plot.plot(
            pen=pg.mkPen(color="#2f5fa0", width=1.2)
        )
        self.playback_cursor = pg.InfiniteLine(
            pos=0.0, angle=90, movable=True, pen=pg.mkPen("#ff8a00", width=2)
        )
        self.playback_cursor.setBounds((0.0, 1.0))
        self.playback_cursor.setZValue(10)
        # 드래그 중 연속 seek 대신 드래그 완료 시점에만 seek 적용
        self.playback_cursor.sigPositionChangeFinished.connect(
            self._on_playback_cursor_moved
        )
        self.recorded_plot.addItem(self.playback_cursor)
        self.playback_cursor.hide()

        plot_layout.addWidget(self.live_plot)
        plot_layout.addWidget(self.recorded_plot)
        plot_layout.setStretch(0, 1)
        plot_layout.setStretch(1, 3)

        studio_layout.addWidget(self.rec_group, 1)
        studio_layout.addWidget(self.plot_group, 3)
        self.main_tabs.addTab(self.studio_tab, "")

        root_layout.addWidget(self.main_tabs, 1)

        self.setCentralWidget(root)

    def _apply_language(self) -> None:
        """현재 언어 기준으로 버튼, 라벨, 플롯 타이틀을 일괄 갱신"""
        self.setWindowTitle(self._t("window_title"))
        self.device_group.setTitle(self._t("group_audio_devices"))
        self.presets_group.setTitle(self._t("group_presets_settings"))
        self.effects_group.setTitle(self._t("group_effects_gain"))
        self.plot_group.setTitle(self._t("group_waveforms"))
        self.rec_group.setTitle(self._t("group_recording"))
        self.main_tabs.setTabText(0, self._t("tab_effects"))
        self.main_tabs.setTabText(1, self._t("tab_studio"))

        self.input_label.setText(self._t("label_input_mic"))
        self.output_label.setText(self._t("label_output_speaker"))
        self.language_label.setText(self._t("label_language"))
        self.process_end_mode_label.setText(self._t("label_process_end_mode"))
        self.process_end_mode_combo.setItemText(0, self._t("process_end_mode_auto"))
        self.process_end_mode_combo.setItemText(1, self._t("process_end_mode_manual"))
        self.refresh_btn.setText(self._t("btn_refresh_devices"))
        self._refresh_stream_toggle_button(force=True)
        self._refresh_theme_toggle_button()

        self.preset_label.setText(self._t("label_preset"))
        self._rebuild_preset_combo()
        self.apply_preset_btn.setText(self._t("btn_apply_preset"))
        self.save_settings_btn.setText(self._t("btn_save_settings"))
        self.load_settings_btn.setText(self._t("btn_load_settings"))
        self.reset_defaults_btn.setText(self._t("btn_reset_defaults"))

        self.live_plot.setTitle(self._t("plot_live_title"))
        self.recorded_plot.setTitle(self._t("plot_recorded_title"))
        self.recorded_plot.setLabel("bottom", self._t("plot_time_label"), units="s")

        self.save_path_label.setText(self._t("label_save_path"))
        self.browse_btn.setText(self._t("btn_browse_file"))
        self.pick_folder_btn.setText(self._t("btn_pick_folder"))
        self.record_mode_label.setText(self._t("label_record_output_mode"))
        self.record_mode_combo.setItemText(0, self._t("record_output_mode_always"))
        self.record_mode_combo.setItemText(
            1, self._t("record_output_mode_mute_while_recording")
        )
        rec_start_text = "시작" if self.current_language == "ko" else "Start"
        rec_stop_text = "정지" if self.current_language == "ko" else "Stop"
        self._set_icon_button(
            self.rec_start_btn,
            QStyle.StandardPixmap.SP_DialogApplyButton,
            rec_start_text,
        )
        self.rec_start_btn.setIcon(self._create_shape_icon("circle", "#ef4444"))
        self._set_icon_button(
            self.rec_stop_btn,
            QStyle.StandardPixmap.SP_DialogSaveButton,
            rec_stop_text,
        )
        self.rec_stop_btn.setIcon(self._create_shape_icon("square", "#ef4444"))
        self._set_icon_button(
            self.stop_playback_btn,
            QStyle.StandardPixmap.SP_MediaStop,
            self._t("btn_stop_playback"),
        )
        self._set_record_button_sizes()
        self.routing_hint.setText(self._t("hint_virtual_routing"))
        self._refresh_device_unassigned_label()

        for key, label in self.slider_title_labels.items():
            label.setText(self._t(key))

        if self.rec_stop_btn.isEnabled():
            self.record_status.setText(self._t("status_recording"))
        elif self.record_status.text().strip() == "":
            self.record_status.setText(self._t("status_no_recording"))

        if self.engine.is_running():
            status = self.engine.last_stream_status
            if status:
                self.stream_label.setText(self._t("status_running_flag", status=status))
            else:
                self.stream_label.setText(
                    self._t("status_running_sr", samplerate=self.engine.sample_rate)
                )
        else:
            self.stream_label.setText(self._t("status_stopped"))

        self._refresh_playback_buttons(force=True)

    def _add_slider(
        self,
        layout: QGridLayout,
        row: int,
        title_key: str,
        minimum: int,
        maximum: int,
        initial: int,
        value_to_text,
        on_value_changed,
        text_to_value=None,
    ) -> QSlider:
        """공통 슬라이더 행(라벨/슬라이더/입력 가능 값 필드)을 생성."""
        label = QLabel(self._t(title_key))
        self.slider_title_labels[title_key] = label
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(initial)
        value_edit = QLineEdit(value_to_text(initial))
        value_edit.setMinimumWidth(84)
        value_edit.setMaximumWidth(100)
        value_edit.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        def _sync_value_text(v: int) -> None:
            if not value_edit.hasFocus():
                value_edit.setText(value_to_text(v))

        slider.valueChanged.connect(_sync_value_text)
        slider.valueChanged.connect(on_value_changed)

        def _apply_input_text() -> None:
            parsed_value = self._parse_slider_input_value(
                text=value_edit.text(),
                current_value=slider.value(),
                slider=slider,
                text_to_value=text_to_value,
            )
            slider.setValue(parsed_value)
            value_edit.setText(value_to_text(slider.value()))

        value_edit.editingFinished.connect(_apply_input_text)

        layout.addWidget(label, row, 0)
        layout.addWidget(slider, row, 1)
        layout.addWidget(value_edit, row, 2)
        return slider

    @staticmethod
    def _extract_number(text: str) -> Optional[float]:
        """문자열에서 첫 번째 숫자 토큰을 추출."""
        if not text:
            return None
        match = re.search(r"[-+]?\d*\.?\d+", text.replace(",", "."))
        if not match:
            return None
        try:
            return float(match.group(0))
        except Exception:
            return None

    def _parse_slider_input_value(
        self,
        text: str,
        current_value: int,
        slider: QSlider,
        text_to_value=None,
    ) -> int:
        """값 입력 필드 문자열을 슬라이더 정수 값으로 파싱/보정."""
        parsed: Optional[float] = None
        if text_to_value is not None:
            try:
                parsed = float(text_to_value(text))
            except Exception:
                parsed = None

        if parsed is None:
            parsed = self._extract_number(text)
        if parsed is None:
            return int(np.clip(current_value, slider.minimum(), slider.maximum()))

        as_int = int(round(parsed))
        return int(np.clip(as_int, slider.minimum(), slider.maximum()))

    def _parse_distortion_input_value(self, text: str) -> int:
        """디스토션 입력값 파싱: 1.50x 또는 150 같은 입력을 모두 허용."""
        value = self._extract_number(text)
        if value is None:
            raise ValueError("invalid number")
        normalized = text.lower().strip()
        if "x" in normalized or value <= 8.0:
            return int(round((value - 1.0) * 100.0))
        return int(round(value))

    # -------- 파라미터/장치 동기화 --------
    def _sync_all_params(self) -> None:
        """현재 슬라이더 값을 엔진 파라미터로 동기화"""
        self.engine.set_param("input_gain", self.input_gain_slider.value() / 100.0)
        self.engine.set_param("output_gain", self.output_gain_slider.value() / 100.0)
        self.engine.set_param(
            "distortion_drive", 1.0 + (self.distortion_slider.value() / 100.0)
        )
        self.engine.set_param("reverb_wet", self.reverb_slider.value() / 100.0)
        self.engine.set_param("delay_wet", self.delay_mix_slider.value() / 100.0)
        self.engine.set_param("delay_time_ms", float(self.delay_time_slider.value()))
        self.engine.set_param(
            "delay_feedback", self.delay_feedback_slider.value() / 100.0
        )
        self.engine.set_param("echo_wet", self.echo_mix_slider.value() / 100.0)
        self.engine.set_param("echo_time_ms", float(self.echo_time_slider.value()))
        self.engine.set_param(
            "echo_feedback", self.echo_feedback_slider.value() / 100.0
        )

    def _load_devices(self) -> None:
        """입출력 장치 목록을 다시 조회하고 콤보 박스를 갱신"""
        current_in = self.input_combo.currentData()
        current_out = self.output_combo.currentData()

        self.input_combo.clear()
        self.output_combo.clear()
        self.input_combo.addItem(self._t("device_unassigned"), None)
        self.output_combo.addItem(self._t("device_unassigned"), None)

        try:
            inputs, outputs = self.engine.list_devices()
        except Exception as exc:
            QMessageBox.critical(self, self._t("dialog_device_error"), str(exc))
            return

        for idx, name in inputs:
            self.input_combo.addItem(f"[{idx}] {name}", idx)
        for idx, name in outputs:
            self.output_combo.addItem(f"[{idx}] {name}", idx)

        default_in, default_out = sd.default.device
        self._restore_combo(self.input_combo, current_in, default_in)
        self._restore_combo(self.output_combo, current_out, default_out)

    @staticmethod
    def _restore_combo(
        combo: QComboBox, previous: Optional[int], fallback: int
    ) -> None:
        """이전 선택값 또는 기본 장치값으로 콤보 선택을 복원"""
        for target in (previous, fallback):
            if target is None:
                continue
            idx = combo.findData(target)
            if idx >= 0:
                combo.setCurrentIndex(idx)
                return
        if combo.count() > 0:
            combo.setCurrentIndex(0)

    def _refresh_device_unassigned_label(self) -> None:
        """장치 콤보의 미할당 항목 텍스트를 현재 언어로 갱신."""
        for combo in (self.input_combo, self.output_combo):
            idx = combo.findData(None)
            if idx >= 0:
                combo.setItemText(idx, self._t("device_unassigned"))

    # -------- 스트림/녹음/재생 제어 --------
    @staticmethod
    def _normalize_record_file_path(path: Path) -> Path:
        """녹음 저장 경로를 wav/m4a 중 지원 확장자로 정규화한다."""
        suffix = path.suffix.lower()
        if suffix in {".wav", ".m4a"}:
            return path
        if suffix:
            return path.with_suffix(".wav")
        return path.with_suffix(".wav")

    @staticmethod
    def _run_subprocess_or_raise(command: List[str]) -> None:
        """외부 명령 실행 실패 시 stderr를 포함한 예외를 발생시킨다."""
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            reason = proc.stderr.strip() or proc.stdout.strip() or "unknown error"
            raise RuntimeError(reason)

    def _get_ffmpeg_executable(self) -> str:
        """m4a 처리에 사용할 ffmpeg 실행 파일 경로를 반환한다."""
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
        except Exception as exc:
            raise RuntimeError(self._t("msg_m4a_dep_missing")) from exc
        try:
            return str(get_ffmpeg_exe())
        except Exception as exc:
            raise RuntimeError(self._t("msg_m4a_dep_missing")) from exc

    def _save_audio_file(
        self, save_path: Path, audio: np.ndarray, samplerate: int
    ) -> None:
        """선택된 확장자에 맞춰 WAV 또는 M4A로 저장한다."""
        suffix = save_path.suffix.lower()
        if suffix == ".m4a":
            ffmpeg = self._get_ffmpeg_executable()
            with tempfile.TemporaryDirectory(prefix="ve_m4a_encode_") as tmp_dir:
                tmp_wav = Path(tmp_dir) / "input.wav"
                sf.write(str(tmp_wav), audio, samplerate, subtype="PCM_16")
                command = [
                    ffmpeg,
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(tmp_wav),
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    str(save_path),
                ]
                try:
                    self._run_subprocess_or_raise(command)
                except Exception as exc:
                    raise RuntimeError(
                        self._t("msg_m4a_convert_failed", reason=str(exc))
                    ) from exc
            return

        sf.write(str(save_path), audio, samplerate, subtype="PCM_16")

    def _load_audio_file(self, source_path: Path) -> Tuple[np.ndarray, int]:
        """파일을 float32 2D 배열(프레임, 채널)로 로드한다."""
        try:
            data, samplerate = sf.read(
                str(source_path), dtype="float32", always_2d=True
            )
            return data, int(samplerate)
        except Exception:
            if source_path.suffix.lower() != ".m4a":
                raise

            ffmpeg = self._get_ffmpeg_executable()
            with tempfile.TemporaryDirectory(prefix="ve_m4a_decode_") as tmp_dir:
                decoded_wav = Path(tmp_dir) / "decoded.wav"
                command = [
                    ffmpeg,
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(source_path),
                    "-vn",
                    "-c:a",
                    "pcm_s16le",
                    str(decoded_wav),
                ]
                try:
                    self._run_subprocess_or_raise(command)
                    data, samplerate = sf.read(
                        str(decoded_wav), dtype="float32", always_2d=True
                    )
                    return data, int(samplerate)
                except Exception as exc:
                    raise RuntimeError(
                        self._t("msg_m4a_convert_failed", reason=str(exc))
                    ) from exc

    def _prepare_playback_audio_for_device(
        self, source_path: Path, output_device: int
    ) -> None:
        """재생 소스를 로드하고 출력 장치 채널 수에 맞춘다."""
        resolved_path = source_path.expanduser().resolve()
        should_reload = (
            self.playback_audio is None
            or self.playback_source_path != resolved_path
            or self.playback_output_device != output_device
        )
        if not should_reload:
            return

        data, samplerate = self._load_audio_file(resolved_path)
        if data.ndim == 1:
            data = data[:, np.newaxis]

        out_info = sd.query_devices(output_device)
        max_channels = int(out_info["max_output_channels"])
        if max_channels < 1:
            raise RuntimeError(self._t("msg_select_output_before_playback"))

        if data.shape[1] > max_channels:
            data = data[:, :max_channels]
        elif data.shape[1] == 1 and max_channels >= 2:
            data = np.repeat(data, 2, axis=1)

        data = np.ascontiguousarray(data.astype(np.float32, copy=False))
        total_frames = int(data.shape[0])
        with self.playback_lock:
            self.playback_audio = data
            self.playback_samplerate = int(samplerate)
            self.playback_total_frames = total_frames
            self.playback_current_frame = 0

        self.playback_source_path = resolved_path
        self.playback_output_device = output_device
        duration_seconds = total_frames / float(samplerate) if samplerate > 0 else 0.0
        self._set_recorded_time_axis(duration_seconds)
        self._set_playback_cursor_seconds(0.0)
        if total_frames > 0:
            self.playback_cursor.show()
        else:
            self.playback_cursor.hide()
        self._refresh_playback_buttons()

    def _is_playback_running(self) -> bool:
        """재생 스트림 활성 상태를 확인한다."""
        stream = self.playback_stream
        if stream is None:
            return False
        try:
            return bool(stream.active)
        except Exception:
            return False

    def _get_playback_status_path(self) -> Optional[Path]:
        """상태 표시용 현재 재생 파일 경로를 반환한다."""
        if self.playback_source_path is not None:
            return self.playback_source_path
        if self.last_recording_path is not None:
            return self.last_recording_path
        maybe_path = Path(self.path_edit.text().strip()).expanduser()
        if maybe_path.exists():
            return maybe_path
        return None

    def _refresh_playback_buttons(self, force: bool = False) -> None:
        """재생/일시정지 아이콘과 정지 버튼 활성 상태를 갱신한다."""
        engine_running = self.engine.is_running()
        if engine_running:
            signature = (True, False, False, self.current_language)
            if (not force) and self.playback_button_signature == signature:
                return
            self.playback_button_signature = signature
            self.play_btn.setEnabled(False)
            self.stop_playback_btn.setEnabled(False)
            self._set_icon_button(
                self.play_btn,
                QStyle.StandardPixmap.SP_MediaPlay,
                self._t("btn_play_pause_play"),
            )
            return

        is_running = self._is_playback_running()
        with self.playback_lock:
            has_position = (
                self.playback_total_frames > 0 and self.playback_current_frame > 0
            )
        signature = (False, bool(is_running), bool(has_position), self.current_language)
        if (not force) and self.playback_button_signature == signature:
            return
        self.playback_button_signature = signature

        self.play_btn.setEnabled(True)
        if is_running:
            self._set_icon_button(
                self.play_btn,
                QStyle.StandardPixmap.SP_MediaPause,
                self._t("btn_play_pause_pause"),
            )
        else:
            self._set_icon_button(
                self.play_btn,
                QStyle.StandardPixmap.SP_MediaPlay,
                self._t("btn_play_pause_play"),
            )
        self.stop_playback_btn.setEnabled(is_running or has_position)

    def _stop_playback_to_start(self) -> None:
        """재생을 정지하고 위치를 0초로 되돌린다."""
        if self._is_playback_running():
            self._stop_playback_stream()
        self._seek_playback_frame(0)
        self._set_playback_cursor_seconds(0.0)
        status_path = self._get_playback_status_path()
        if status_path is not None:
            self.record_status.setText(
                self._t("status_playback_stopped", path=status_path)
            )
        self._refresh_playback_buttons()

    def _stop_playback_stream(self, clear_audio: bool = False) -> None:
        """재생 스트림을 정지하고 필요하면 재생 버퍼까지 초기화한다."""
        stream = self.playback_stream
        self.playback_stream = None
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

        if clear_audio:
            with self.playback_lock:
                self.playback_audio = None
                self.playback_samplerate = 0
                self.playback_total_frames = 0
                self.playback_current_frame = 0
            self.playback_source_path = None
            self.playback_output_device = None
            self._set_recorded_time_axis(1.0)
            self._set_playback_cursor_seconds(0.0)
            self.playback_cursor.hide()

        if hasattr(self, "play_btn"):
            self._refresh_playback_buttons()

    def _seek_playback_frame(self, frame: int) -> None:
        """재생 커서를 지정 프레임으로 이동한다."""
        with self.playback_lock:
            total = self.playback_total_frames
            if total <= 0:
                self.playback_current_frame = 0
                return
            self.playback_current_frame = int(np.clip(frame, 0, total))

    def _set_recorded_time_axis(self, duration_seconds: float) -> None:
        """녹음 파형 X축을 항상 0초부터 시작하도록 고정한다."""
        max_time = max(float(duration_seconds), 0.01)
        self.recorded_plot.setLimits(xMin=0.0, xMax=max_time, yMin=-2.0, yMax=2.0)
        self.recorded_plot.setXRange(0.0, max_time, padding=0.0)
        self.recorded_plot.setYRange(-2.0, 2.0, padding=0.0)
        self.playback_cursor.setBounds((0.0, max_time))

    def _set_playback_cursor_seconds(self, seconds: float) -> None:
        """파형 위 세로 막대(|)를 지정 시간 위치로 이동한다."""
        lower, upper = self.playback_cursor.bounds()
        clamped = float(np.clip(seconds, lower, upper))
        self.playback_cursor_internal_update = True
        self.playback_cursor.setValue(clamped)
        self.playback_cursor_internal_update = False

    def _get_playback_cursor_seconds(self) -> float:
        """파형 위 세로 막대(|)의 현재 시간을 반환한다."""
        return max(0.0, float(self.playback_cursor.value()))

    def _on_playback_cursor_moved(self) -> None:
        """사용자가 세로 막대를 움직이면 해당 지점으로 재생 위치를 이동한다."""
        if self.playback_cursor_internal_update:
            return
        with self.playback_lock:
            total_frames = self.playback_total_frames
            samplerate = self.playback_samplerate
        if total_frames <= 0 or samplerate <= 0:
            return
        target_seconds = self._get_playback_cursor_seconds()
        target_frame = int(round(target_seconds * float(samplerate)))
        self._seek_playback_frame(target_frame)
        self._refresh_playback_buttons()

    def _playback_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info,
        status: sd.CallbackFlags,
    ) -> None:
        """재생 커서 기준으로 오디오 블록을 출력 버퍼에 채운다."""
        _ = time_info
        _ = status
        with self.playback_lock:
            audio = self.playback_audio
            total = self.playback_total_frames
            cursor = self.playback_current_frame
            if audio is None or total <= 0:
                outdata.fill(0.0)
                raise sd.CallbackStop()

            start = int(np.clip(cursor, 0, total))
            end = min(start + frames, total)
            chunk = audio[start:end]
            self.playback_current_frame = end
            is_finished = end >= total

        outdata.fill(0.0)
        if chunk.size > 0:
            outdata[: chunk.shape[0], : chunk.shape[1]] = chunk
        if is_finished:
            raise sd.CallbackStop()

    def _sync_playback_cursor_line(self) -> None:
        """재생 중 현재 프레임을 파형 위 세로 막대(|)로 동기화한다."""
        if not self._is_playback_running():
            return
        with self.playback_lock:
            total = self.playback_total_frames
            frame = self.playback_current_frame
            samplerate = self.playback_samplerate

        if total <= 0 or samplerate <= 0:
            return

        frame = int(np.clip(frame, 0, total))
        if frame == self.last_cursor_frame_synced:
            return
        self.last_cursor_frame_synced = frame
        seconds = frame / float(samplerate)
        self._set_playback_cursor_seconds(seconds)

    def _start_stream(self) -> None:
        """선택 장치로 스트림을 시작하고 UI 상태를 실행 중으로 전환"""
        input_dev = self.input_combo.currentData()
        output_dev = self.output_combo.currentData()
        if input_dev is None or output_dev is None:
            QMessageBox.warning(
                self,
                self._t("dialog_missing_device"),
                self._t("msg_select_input_output"),
            )
            return

        try:
            self._stop_playback_stream()
            self.engine.set_devices(int(input_dev), int(output_dev))
            samplerate = self.engine.start()
        except Exception as exc:
            QMessageBox.critical(self, self._t("dialog_audio_start_error"), str(exc))
            return

        self.stream_label.setText(self._t("status_running_sr", samplerate=samplerate))
        self.stream_expected_running = True
        self._refresh_stream_toggle_button()
        self._refresh_playback_buttons()

    def _stop_stream(self) -> None:
        """스트림을 중지하고 UI 상태를 대기 상태로 복귀시킨다."""
        try:
            self.engine.stop()
        except Exception as exc:
            QMessageBox.warning(self, self._t("dialog_audio_stop_warning"), str(exc))
        self.stream_expected_running = False
        self.stream_label.setText(self._t("status_stopped"))
        self._refresh_stream_toggle_button()
        self.rec_start_btn.setEnabled(True)
        self.rec_stop_btn.setEnabled(False)
        self._refresh_playback_buttons()

    def _browse_record_file(self) -> None:
        """사용자가 저장할 WAV 파일 경로를 직접 선택가능하게 한다."""
        current = self.path_edit.text().strip() or str(Path.cwd() / "recording.wav")
        path, selected_filter = QFileDialog.getSaveFileName(
            self, self._t("dialog_save_as"), current, self._t("filter_audio_save")
        )
        if path:
            selected_path = Path(path).expanduser()
            if selected_path.suffix.lower() not in {".wav", ".m4a"}:
                if "m4a" in selected_filter.lower():
                    selected_path = selected_path.with_suffix(".m4a")
                else:
                    selected_path = selected_path.with_suffix(".wav")
            normalized = self._normalize_record_file_path(selected_path)
            self.path_edit.setText(str(normalized))

    def _select_save_folder(self) -> None:
        """저장 폴더를 선택하고 파일명은 기존 값 또는 기본값으로 유지"""
        current_text = self.path_edit.text().strip() or str(
            Path.cwd() / "recording.wav"
        )
        current_path = Path(current_text).expanduser()
        default_dir = str(current_path.parent if current_path.suffix else current_path)
        folder = QFileDialog.getExistingDirectory(
            self, self._t("dialog_select_folder"), default_dir
        )
        if not folder:
            return

        filename = (
            current_path.name
            if current_path.suffix.lower() in {".wav", ".m4a"}
            else "recording.wav"
        )
        self.path_edit.setText(str(Path(folder) / filename))

    def _start_recording(self) -> None:
        """스트림 실행 상태를 확인한 뒤 녹음을 시작"""
        if not self.engine.is_running():
            QMessageBox.warning(
                self,
                self._t("dialog_stream_not_running"),
                self._t("msg_start_processing_first"),
            )
            return

        # 재녹음 시에도 선택 모드가 누락되지 않도록 시작 시점에 모드를 재적용.
        self._stop_playback_stream()
        self.engine.start_recording(self._get_selected_record_output_mode())
        self.record_status.setText(self._t("status_recording"))
        self.rec_start_btn.setEnabled(False)
        self.rec_stop_btn.setEnabled(True)

    def _stop_recording(self) -> None:
        """녹음을 종료하고 WAV 파일로 저장한 뒤 파형을 업데이트"""
        audio, samplerate = self.engine.stop_recording()
        self.rec_start_btn.setEnabled(True)
        self.rec_stop_btn.setEnabled(False)
        should_auto_end = self.process_end_mode == "auto"

        if audio.size == 0:
            self.record_status.setText(self._t("status_no_audio_captured"))
            if should_auto_end and self.engine.is_running():
                self._stop_stream()
            return

        save_path_text = self.path_edit.text().strip()
        if save_path_text:
            save_path = Path(save_path_text).expanduser()
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = Path.cwd() / f"recording_{stamp}.wav"

        if save_path.exists() and save_path.is_dir():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = save_path / f"recording_{stamp}.wav"
        save_path = self._normalize_record_file_path(save_path)

        self.path_edit.setText(str(save_path))
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._save_audio_file(save_path, audio, samplerate)
        except Exception as exc:
            QMessageBox.critical(self, self._t("dialog_save_error"), str(exc))
            return

        self._stop_playback_stream(clear_audio=True)
        self.last_recording_path = save_path
        total_frames = int(audio.shape[0])
        with self.playback_lock:
            self.playback_total_frames = total_frames
            self.playback_samplerate = int(samplerate)
            self.playback_current_frame = 0
        self.playback_source_path = save_path.resolve()
        self.playback_output_device = None
        self._set_playback_cursor_seconds(0.0)
        self._refresh_playback_buttons()

        duration = len(audio) / float(samplerate)
        self.record_status.setText(
            self._t("status_saved", path=save_path, duration=duration)
        )
        self._plot_recorded(audio, samplerate)

        if should_auto_end and self.engine.is_running():
            self._stop_stream()

    def _play_recording(self) -> None:
        """저장된 녹음 파일을 선택된 출력 장치로 재생"""
        if self._is_playback_running():
            self._stop_playback_stream()
            status_path = self._get_playback_status_path()
            if status_path is not None:
                self.record_status.setText(
                    self._t("status_playback_paused", path=status_path)
                )
            self._refresh_playback_buttons()
            return

        if self.last_recording_path is None:
            maybe_path = Path(self.path_edit.text().strip()).expanduser()
            if maybe_path.exists():
                self.last_recording_path = maybe_path
            else:
                QMessageBox.information(
                    self, self._t("dialog_no_recording"), self._t("msg_record_first")
                )
                return

        if self.engine.is_running():
            QMessageBox.information(
                self,
                self._t("dialog_processing_active"),
                self._t("msg_playback_while_processing_locked"),
            )
            return

        output_dev = self.output_combo.currentData()
        if output_dev is None:
            QMessageBox.warning(
                self,
                self._t("dialog_missing_device"),
                self._t("msg_select_output_before_playback"),
            )
            return

        try:
            target_seconds = self._get_playback_cursor_seconds()
            self._prepare_playback_audio_for_device(
                self.last_recording_path, int(output_dev)
            )

            with self.playback_lock:
                audio = self.playback_audio
                samplerate = self.playback_samplerate
                total = self.playback_total_frames
                if audio is None or total <= 0:
                    raise RuntimeError(self._t("msg_record_first"))
                target_frame = int(round(target_seconds * float(samplerate)))
                target_frame = int(np.clip(target_frame, 0, total))
                if target_frame >= total:
                    target_frame = 0
                self.playback_current_frame = target_frame
                channels = int(audio.shape[1])

            self._plot_recorded(audio[:, 0], samplerate)
            self._set_playback_cursor_seconds(target_frame / float(samplerate))

            self.playback_stream = sd.OutputStream(
                samplerate=samplerate,
                blocksize=self.engine.block_size,
                latency="high",
                device=int(output_dev),
                channels=channels,
                dtype="float32",
                callback=self._playback_callback,
            )
            self.playback_stream.start()
            self.record_status.setText(
                self._t("status_playing", path=self.last_recording_path)
            )
            self._refresh_playback_buttons()
        except Exception as exc:
            self._stop_playback_stream()
            QMessageBox.critical(self, self._t("dialog_playback_error"), str(exc))

    # -------- 실시간 표시/종료 처리 --------
    def _update_live_waveform(self) -> None:
        """타이머 주기로 실시간 파형/스트림 상태를 갱신"""
        studio_visible = (
            hasattr(self, "main_tabs")
            and self.main_tabs.currentIndex() == 1
            and self.isVisible()
            and self.isActiveWindow()
        )
        if studio_visible:
            if self.engine.is_running():
                samples = self.engine.get_latest_output()
                if samples.size > 0:
                    self.live_curve.setData(samples)
                    self.live_wave_visible = True
            elif self.live_wave_visible:
                # 스트림 중지 상태에서는 실시간 파형 갱신을 멈춰 UI 부하를 줄인다.
                self.live_curve.setData([])
                self.live_wave_visible = False
            self._sync_playback_cursor_line()
        if self.playback_stream is not None and not self._is_playback_running():
            self._stop_playback_stream()
            if self.playback_source_path is not None:
                self.record_status.setText(
                    self._t("status_playback_done", path=self.playback_source_path)
                )

        status = self.engine.last_stream_status
        if self.engine.is_running():
            self._refresh_stream_toggle_button()
            if status:
                self.stream_label.setText(self._t("status_running_flag", status=status))
            return

        # UI는 실행중인데 스트림이 꺼졌다면 예외/장치 오류로 판단하고 상태를 해제한다.
        if self.stream_expected_running:
            self._stop_stream()
            if status:
                self.stream_label.setText(
                    self._t("status_stopped_error", status=status)
                )
                QMessageBox.warning(self, self._t("dialog_stream_error"), status)
        else:
            self._refresh_stream_toggle_button()

    def _plot_recorded(self, audio: np.ndarray, samplerate: int) -> None:
        """녹음 데이터를 다운샘플링해 기록 파형 플롯에 그린다."""
        if audio.ndim > 1:
            audio = audio[:, 0]
        if audio.size == 0 or samplerate <= 0:
            self.recorded_curve.setData([])
            self._set_recorded_time_axis(1.0)
            self.playback_cursor.hide()
            return

        duration = len(audio) / float(samplerate)
        # 매우 긴 녹음에서도 플로팅 부하를 줄이기 위해 표시 샘플 수 제한
        step = max(1, audio.size // 12000)
        trimmed = audio[::step]
        times = np.arange(trimmed.size, dtype=np.float64) * (step / float(samplerate))
        self.recorded_curve.setData(times, trimmed)
        self.recorded_plot.setLabel("bottom", self._t("plot_time_label"), units="s")
        self._set_recorded_time_axis(duration)
        self.playback_cursor.show()
        self._sync_playback_cursor_line()

    def closeEvent(self, event) -> None:
        """창 종료 시 오디오 스트림과 재생을 정리"""
        try:
            self._stop_playback_stream(clear_audio=True)
            self.engine.stop()
            sd.stop()
        except Exception:
            pass
        super().closeEvent(event)


# ============================================================
# 런타임 유틸리티 (리소스 경로/아이콘/윈도우 설정)
# ============================================================
def _runtime_base_dir() -> Path:
    """스크립트 실행/패키징 실행(PyInstaller) 모두에서 기준 경로를 반환"""
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _find_app_icon() -> Optional[Path]:
    """assets/icons에서 우선순위에 맞는 앱 아이콘 파일을 탐색"""
    icon_dir = _runtime_base_dir() / "assets" / "icons"
    # 우선순위: app.ico -> 임의 .ico -> app.png -> 기타 이미지
    app_ico = icon_dir / "app.ico"
    if app_ico.exists():
        return app_ico

    ico_files = sorted(
        [
            path
            for path in icon_dir.glob("*.ico")
            if path.name.lower() not in {".gitkeep", "thumbs.db"}
        ]
    )
    if ico_files:
        return ico_files[0]

    preferred_names = ("app.png",)
    for name in preferred_names:
        candidate = icon_dir / name
        if candidate.exists():
            return candidate

    image_patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    for pattern in image_patterns:
        candidates = sorted(
            [
                path
                for path in icon_dir.glob(pattern)
                if path.name.lower() not in {".gitkeep", "thumbs.db"}
            ]
        )
        if candidates:
            return candidates[0]
    return None


def _set_windows_app_user_model_id() -> None:
    """윈도우 작업표시줄 아이콘을 AppUserModelID로 설정"""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "voiceeffect.studio.app"
        )
    except Exception:
        pass


# ============================================================
# 애플리케이션 진입점
# ============================================================
def main() -> None:
    """QApplication을 생성하고 아이콘/메인윈도우를 초기화해 실행"""
    _set_windows_app_user_model_id()
    app = QApplication(sys.argv)
    icon_path = _find_app_icon()

    if icon_path is not None:
        icon = QIcon(str(icon_path))
        if not icon.isNull():
            app.setWindowIcon(icon)

    window = MainWindow()
    if icon_path is not None:
        icon = QIcon(str(icon_path))
        if not icon.isNull():
            window.setWindowIcon(icon)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
