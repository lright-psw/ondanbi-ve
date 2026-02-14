import copy
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
import sounddevice as sd
import soundfile as sf
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
    QComboBox,
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
        self.block_size = 1024
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
            "label_input_mic": "입력 마이크",
            "label_output_speaker": "출력 스피커",
            "label_language": "언어",
            "label_process_end_mode": "처리 종료 모드",
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
            "btn_browse_file": "파일 선택",
            "btn_pick_folder": "폴더 선택",
            "record_output_mode_always": "실시간 출력",
            "record_output_mode_mute_while_recording": "녹음 중 무음 (재생 시 출력)",
            "btn_start_recording": "녹음 시작",
            "btn_stop_and_save": "중지 후 저장",
            "btn_play_last": "최근 녹음 재생",
            "status_no_recording": "아직 녹음된 파일이 없습니다.",
            "status_recording": "녹음 중...",
            "status_no_audio_captured": "녹음을 중지했지만 캡처된 오디오가 없습니다.",
            "status_saved": "저장 완료: {path} ({duration:.2f}초)",
            "status_playing": "재생 중: {path}",
            "hint_virtual_routing": "팁: Discord/음성 녹음기 등 외부 앱은 가상 오디오 케이블 장치로 라우팅하세요.",
            "dialog_device_error": "장치 오류",
            "dialog_missing_device": "장치 선택 필요",
            "msg_select_input_output": "입력과 출력을 모두 선택하세요.",
            "dialog_audio_start_error": "오디오 시작 오류",
            "dialog_audio_stop_warning": "오디오 중지 경고",
            "dialog_save_as": "녹음 파일로 저장",
            "dialog_select_folder": "저장 폴더 선택",
            "filter_wav": "WAV 파일 (*.wav)",
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
        },
        "en": {
            "window_title": "Ondanbi VE Studio",
            "group_audio_devices": "Audio Devices",
            "group_effects_gain": "Effects and Gain",
            "group_waveforms": "Waveforms",
            "group_recording": "Recording",
            "label_input_mic": "Input Mic",
            "label_output_speaker": "Output Speaker",
            "label_language": "Language",
            "label_process_end_mode": "Process End Mode",
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
            "btn_browse_file": "Browse File",
            "btn_pick_folder": "Choose Folder",
            "record_output_mode_always": "Live Monitor",
            "record_output_mode_mute_while_recording": "Mute While Recording (playback only)",
            "btn_start_recording": "Start Recording",
            "btn_stop_and_save": "Stop and Save",
            "btn_play_last": "Play Last Recording",
            "status_no_recording": "No recording yet.",
            "status_recording": "Recording...",
            "status_no_audio_captured": "Recording stopped (no audio captured).",
            "status_saved": "Saved: {path} ({duration:.2f} sec)",
            "status_playing": "Playing: {path}",
            "hint_virtual_routing": "Tip: For external apps (Discord/Voice Recorder), route output to a virtual audio cable device.",
            "dialog_device_error": "Device Error",
            "dialog_missing_device": "Missing Device",
            "msg_select_input_output": "Select both input and output.",
            "dialog_audio_start_error": "Audio Start Error",
            "dialog_audio_stop_warning": "Audio Stop Warning",
            "dialog_save_as": "Save Recording As",
            "dialog_select_folder": "Select Save Folder",
            "filter_wav": "WAV files (*.wav)",
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
        },
    }

    def __init__(self) -> None:
        """엔진/상태를 초기화하고 UI를 구성한 뒤 타이머 갱신을 시작"""
        super().__init__()
        self.resize(1180, 860)
        pg.setConfigOptions(antialias=True)

        self.engine = AudioEngine()
        self.last_recording_path: Optional[Path] = None
        self.current_language = "ko"
        self.slider_title_labels: dict[str, QLabel] = {}
        self.process_end_mode = "auto"

        self._build_ui()
        self._apply_language()
        self._load_devices()
        self._sync_all_params()

        self.wave_timer = QTimer(self)
        self.wave_timer.setInterval(40)
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

    # -------- UI 구성 --------
    def _build_ui(self) -> None:
        """장치/이펙트/파형/녹음 UI를 생성하고 시그널을 연결"""
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setSpacing(10)

        # 1) 장치/언어/처리 시작-중지 제어 영역
        self.device_group = QGroupBox()
        device_layout = QGridLayout(self.device_group)
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
        self.start_btn.clicked.connect(self._start_stream)
        self.stop_btn = QPushButton()
        self.stop_btn.clicked.connect(self._stop_stream)
        self.stop_btn.setEnabled(False)
        self.stream_label = QLabel()

        device_layout.addWidget(self.input_label, 0, 0)
        device_layout.addWidget(self.input_combo, 0, 1)
        device_layout.addWidget(self.output_label, 1, 0)
        device_layout.addWidget(self.output_combo, 1, 1)
        device_layout.addWidget(self.language_label, 2, 0)
        device_layout.addWidget(self.language_combo, 2, 1)
        device_layout.addWidget(self.process_end_mode_label, 3, 0)
        device_layout.addWidget(self.process_end_mode_combo, 3, 1)
        device_layout.addWidget(self.refresh_btn, 0, 2)
        device_layout.addWidget(self.start_btn, 1, 2)
        device_layout.addWidget(self.stop_btn, 1, 3)
        device_layout.addWidget(self.stream_label, 0, 3)
        root_layout.addWidget(self.device_group)

        # 2) 이펙트 및 게인 조절 영역
        self.effects_group = QGroupBox()
        effects_layout = QGridLayout(self.effects_group)
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
            100,
            800,
            100,
            lambda v: f"{v / 100.0:.2f}x",
            lambda v: self.engine.set_param("distortion_drive", v / 100.0),
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
            20,
            1500,
            280,
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
            35,
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
            100,
            2500,
            150,
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
            30,
            lambda v: f"{v}%",
            lambda v: self.engine.set_param("echo_feedback", v / 100.0),
        )
        root_layout.addWidget(self.effects_group)

        # 3) 실시간/녹음 파형 표시 영역
        self.plot_group = QGroupBox()
        plot_layout = QVBoxLayout(self.plot_group)
        self.live_plot = pg.PlotWidget()
        self.live_plot.setYRange(-1.0, 1.0)
        self.live_plot.showGrid(x=True, y=True, alpha=0.22)
        self.live_curve = self.live_plot.plot(pen=pg.mkPen(color="#3f9f5f", width=1.5))

        self.recorded_plot = pg.PlotWidget()
        self.recorded_plot.setYRange(-1.0, 1.0)
        self.recorded_plot.showGrid(x=True, y=True, alpha=0.22)
        self.recorded_curve = self.recorded_plot.plot(
            pen=pg.mkPen(color="#2f5fa0", width=1.2)
        )

        plot_layout.addWidget(self.live_plot)
        plot_layout.addWidget(self.recorded_plot)
        root_layout.addWidget(self.plot_group, 1)

        # 4) 녹음/저장/재생 및 모니터링 모드 영역
        self.rec_group = QGroupBox()
        rec_layout = QGridLayout(self.rec_group)
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
        # UI 기본 선택  "녹음 중 무음"
        self.record_mode_combo.setCurrentIndex(1)
        self.engine.set_record_output_mode(self._get_selected_record_output_mode())

        self.rec_start_btn = QPushButton()
        self.rec_start_btn.clicked.connect(self._start_recording)
        self.rec_stop_btn = QPushButton()
        self.rec_stop_btn.clicked.connect(self._stop_recording)
        self.rec_stop_btn.setEnabled(False)
        self.play_btn = QPushButton()
        self.play_btn.clicked.connect(self._play_recording)

        self.record_status = QLabel()
        self.routing_hint = QLabel()

        rec_layout.addWidget(self.save_path_label, 0, 0)
        rec_layout.addWidget(self.path_edit, 0, 1, 1, 2)
        rec_layout.addWidget(self.browse_btn, 0, 3)
        rec_layout.addWidget(self.pick_folder_btn, 0, 4)
        rec_layout.addWidget(self.rec_start_btn, 1, 1)
        rec_layout.addWidget(self.rec_stop_btn, 1, 2)
        rec_layout.addWidget(self.play_btn, 1, 3)
        rec_layout.addWidget(self.record_mode_label, 2, 0)
        rec_layout.addWidget(self.record_mode_combo, 2, 1, 1, 3)
        rec_layout.addWidget(self.record_status, 3, 1, 1, 4)
        rec_layout.addWidget(self.routing_hint, 4, 1, 1, 4)
        root_layout.addWidget(self.rec_group)

        self.setCentralWidget(root)

    def _apply_language(self) -> None:
        """현재 언어 기준으로 버튼, 라벨, 플롯 타이틀을 일괄 갱신"""
        self.setWindowTitle(self._t("window_title"))
        self.device_group.setTitle(self._t("group_audio_devices"))
        self.effects_group.setTitle(self._t("group_effects_gain"))
        self.plot_group.setTitle(self._t("group_waveforms"))
        self.rec_group.setTitle(self._t("group_recording"))

        self.input_label.setText(self._t("label_input_mic"))
        self.output_label.setText(self._t("label_output_speaker"))
        self.language_label.setText(self._t("label_language"))
        self.process_end_mode_label.setText(self._t("label_process_end_mode"))
        self.process_end_mode_combo.setItemText(0, self._t("process_end_mode_auto"))
        self.process_end_mode_combo.setItemText(1, self._t("process_end_mode_manual"))
        self.refresh_btn.setText(self._t("btn_refresh_devices"))
        self.start_btn.setText(self._t("btn_start_processing"))
        self.stop_btn.setText(self._t("btn_stop_processing"))

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
        self.rec_start_btn.setText(self._t("btn_start_recording"))
        self.rec_stop_btn.setText(self._t("btn_stop_and_save"))
        self.play_btn.setText(self._t("btn_play_last"))
        self.routing_hint.setText(self._t("hint_virtual_routing"))

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
    ) -> QSlider:
        """공통 슬라이더 행(라벨/슬라이더/값 표시)을 생성해 레이아웃에 추가"""
        label = QLabel(self._t(title_key))
        self.slider_title_labels[title_key] = label
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(initial)
        value_label = QLabel(value_to_text(initial))
        value_label.setMinimumWidth(70)

        slider.valueChanged.connect(lambda v: value_label.setText(value_to_text(v)))
        slider.valueChanged.connect(on_value_changed)

        layout.addWidget(label, row, 0)
        layout.addWidget(slider, row, 1)
        layout.addWidget(value_label, row, 2)
        return slider

    # -------- 파라미터/장치 동기화 --------
    def _sync_all_params(self) -> None:
        """현재 슬라이더 값을 엔진 파라미터로 동기화"""
        self.engine.set_param("input_gain", self.input_gain_slider.value() / 100.0)
        self.engine.set_param("output_gain", self.output_gain_slider.value() / 100.0)
        self.engine.set_param(
            "distortion_drive", self.distortion_slider.value() / 100.0
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

    # -------- 스트림/녹음/재생 제어 --------
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
            self.engine.set_devices(int(input_dev), int(output_dev))
            samplerate = self.engine.start()
        except Exception as exc:
            QMessageBox.critical(self, self._t("dialog_audio_start_error"), str(exc))
            return

        self.stream_label.setText(self._t("status_running_sr", samplerate=samplerate))
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.play_btn.setEnabled(False)

    def _stop_stream(self) -> None:
        """스트림을 중지하고 UI 상태를 대기 상태로 복귀시킨다."""
        try:
            self.engine.stop()
        except Exception as exc:
            QMessageBox.warning(self, self._t("dialog_audio_stop_warning"), str(exc))
        self.stream_label.setText(self._t("status_stopped"))
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.play_btn.setEnabled(True)
        self.rec_start_btn.setEnabled(True)
        self.rec_stop_btn.setEnabled(False)

    def _browse_record_file(self) -> None:
        """사용자가 저장할 WAV 파일 경로를 직접 선택가능하게 한다."""
        current = self.path_edit.text().strip() or str(Path.cwd() / "recording.wav")
        path, _ = QFileDialog.getSaveFileName(
            self, self._t("dialog_save_as"), current, self._t("filter_wav")
        )
        if path:
            if not path.lower().endswith(".wav"):
                path += ".wav"
            self.path_edit.setText(path)

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
            if current_path.suffix.lower() == ".wav"
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
        elif save_path.suffix.lower() != ".wav":
            save_path = save_path.with_suffix(".wav")

        self.path_edit.setText(str(save_path))
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            sf.write(str(save_path), audio, samplerate, subtype="PCM_16")
        except Exception as exc:
            QMessageBox.critical(self, self._t("dialog_save_error"), str(exc))
            return

        self.last_recording_path = save_path
        duration = len(audio) / float(samplerate)
        self.record_status.setText(
            self._t("status_saved", path=save_path, duration=duration)
        )
        self._plot_recorded(audio, samplerate)

        if should_auto_end and self.engine.is_running():
            self._stop_stream()

    def _play_recording(self) -> None:
        """저장된 녹음 파일을 선택된 출력 장치로 재생"""
        if self.last_recording_path is None:
            maybe_path = Path(self.path_edit.text().strip())
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
            data, samplerate = sf.read(str(self.last_recording_path), dtype="float32")
            if data.ndim == 1:
                out_info = sd.query_devices(int(output_dev))
                if int(out_info["max_output_channels"]) >= 2:
                    data = np.column_stack((data, data))
            sd.play(data, samplerate=samplerate, device=int(output_dev))
            self.record_status.setText(
                self._t("status_playing", path=self.last_recording_path)
            )
        except Exception as exc:
            QMessageBox.critical(self, self._t("dialog_playback_error"), str(exc))

    # -------- 실시간 표시/종료 처리 --------
    def _update_live_waveform(self) -> None:
        """타이머 주기로 실시간 파형/스트림 상태를 갱신"""
        samples = self.engine.get_latest_output()
        if samples.size > 0:
            self.live_curve.setData(samples)
        status = self.engine.last_stream_status
        if self.engine.is_running():
            if status:
                self.stream_label.setText(self._t("status_running_flag", status=status))
            return

        # UI는 실행중인데 스트림이 꺼졌다면 예외/장치 오류로 판단하고 상태를 해제한다.
        if not self.start_btn.isEnabled():
            self._stop_stream()
            if status:
                self.stream_label.setText(
                    self._t("status_stopped_error", status=status)
                )
                QMessageBox.warning(self, self._t("dialog_stream_error"), status)

    def _plot_recorded(self, audio: np.ndarray, samplerate: int) -> None:
        """녹음 데이터를 다운샘플링해 기록 파형 플롯에 그린다."""
        if audio.ndim > 1:
            audio = audio[:, 0]
        if audio.size == 0:
            self.recorded_curve.setData([])
            return

        step = max(1, audio.size // 12000)
        trimmed = audio[::step]
        times = np.linspace(0.0, len(audio) / samplerate, num=len(trimmed))
        self.recorded_curve.setData(times, trimmed)
        self.recorded_plot.setLabel("bottom", self._t("plot_time_label"), units="s")

    def closeEvent(self, event) -> None:
        """창 종료 시 오디오 스트림과 재생을 정리"""
        try:
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
