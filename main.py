import copy
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
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
        # 기본값은 저지연과 안정성을 함께 고려한 균형 설정.
        self.block_size = 256
        self.input_channels = 1
        self.output_channels = 1
        self.input_device: Optional[int] = None
        self.output_device: Optional[int] = None
        self.stream_profile = "default"

        self.latest_output = np.zeros(self.block_size, dtype=np.float32)
        self.last_stream_status = ""
        self.last_stream_status_at = 0.0

        self.is_recording = False
        self.recorded_chunks: List[np.ndarray] = []
        self.recorded_mic_chunks: List[np.ndarray] = []
        self.recorded_karaoke_chunks: List[np.ndarray] = []
        self.recording_alignment_frames = 0
        self.recording_sync_offset_ms = 110
        # 기본값: 녹음 중에는 스피커로 내보내지 않고 파일에만 저장.
        self.record_output_mode = "mute_while_recording"
        self.active_record_output_mode = self.record_output_mode

        # 카라오케(MR) 소스/재생 상태
        self.karaoke_source_audio: Optional[np.ndarray] = None
        self.karaoke_source_sr = 0
        self.karaoke_audio: Optional[np.ndarray] = None
        self.karaoke_position = 0
        self.karaoke_playing = False
        self.karaoke_gain = 1.0
        self.karaoke_finished = False
        self.karaoke_sync_frames = 0
        self.input_latency_frames = 0

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

    # -------- 카라오케 트랙 --------
    @staticmethod
    def _resample_audio(data: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        """샘플레이트가 다를 때 선형 보간으로 오디오를 리샘플링"""
        if src_sr <= 0 or dst_sr <= 0 or src_sr == dst_sr:
            return data.astype(np.float32, copy=False)
        if data.shape[0] == 0:
            return np.zeros((0, data.shape[1]), dtype=np.float32)

        target_frames = max(1, int(round(data.shape[0] * (dst_sr / float(src_sr)))))
        if target_frames == data.shape[0]:
            return data.astype(np.float32, copy=False)

        src_x = np.linspace(0.0, 1.0, num=data.shape[0], endpoint=False, dtype=np.float64)
        dst_x = np.linspace(
            0.0, 1.0, num=target_frames, endpoint=False, dtype=np.float64
        )
        resampled = np.empty((target_frames, data.shape[1]), dtype=np.float32)
        for ch in range(data.shape[1]):
            resampled[:, ch] = np.interp(dst_x, src_x, data[:, ch]).astype(np.float32)
        return resampled

    def _update_karaoke_sync_frames(self, stream_latency: Any = None) -> None:
        """입력 경로 지연을 기준으로 MR 싱크 보정 프레임 수를 갱신"""
        input_latency_s = 0.0
        try:
            if isinstance(stream_latency, (tuple, list)) and len(stream_latency) >= 1:
                input_latency_s = float(stream_latency[0])
            elif isinstance(stream_latency, (int, float)):
                # 단일 값만 있으면 입력/출력 합으로 간주하고 절반만 사용
                input_latency_s = float(stream_latency) * 0.5
        except Exception:
            input_latency_s = 0.0

        if input_latency_s <= 0.0 and self.input_device is not None:
            try:
                in_dev = sd.query_devices(self.input_device)
                input_latency_s = float(in_dev.get("default_low_input_latency") or 0.0)
            except Exception:
                input_latency_s = 0.0

        if self.sample_rate <= 0:
            self.karaoke_sync_frames = 0
            self.input_latency_frames = 0
            return

        input_frames = int(round(max(0.0, input_latency_s) * float(self.sample_rate)))
        self.input_latency_frames = int(
            np.clip(input_frames, 0, int(0.25 * float(self.sample_rate)))
        )
        sync_frames = int(self.input_latency_frames)
        # 콜백 1블록 정도를 추가 보정해 체감 싱크를 맞춘다.
        sync_frames += int(self.block_size)
        self.karaoke_sync_frames = int(
            np.clip(sync_frames, 0, int(0.35 * float(self.sample_rate)))
        )

    def _prepare_karaoke_audio_locked(self, keep_progress: bool = True) -> None:
        """현재 스트림(sample rate / output channels)에 맞춰 MR 버퍼를 준비"""
        if (
            self.karaoke_source_audio is None
            or self.karaoke_source_sr <= 0
            or self.output_channels <= 0
        ):
            self.karaoke_audio = None
            self.karaoke_position = 0
            self.karaoke_playing = False
            self.karaoke_finished = False
            return

        source = self.karaoke_source_audio
        previous_progress = 0.0
        sync_frames = int(max(0, self.karaoke_sync_frames))
        if (
            keep_progress
            and self.karaoke_audio is not None
            and self.karaoke_audio.shape[0] > 0
        ):
            effective_prev = int(
                np.clip(
                    self.karaoke_position - sync_frames,
                    0,
                    self.karaoke_audio.shape[0],
                )
            )
            previous_progress = effective_prev / float(self.karaoke_audio.shape[0])

        prepared = self._resample_audio(source, self.karaoke_source_sr, self.sample_rate)
        if prepared.ndim == 1:
            prepared = prepared[:, np.newaxis]

        if self.output_channels == 1:
            if prepared.shape[1] > 1:
                prepared = np.mean(prepared, axis=1, keepdims=True).astype(np.float32)
            else:
                prepared = prepared[:, :1]
        else:
            if prepared.shape[1] == 1:
                prepared = np.repeat(prepared, self.output_channels, axis=1)
            elif prepared.shape[1] > self.output_channels:
                prepared = prepared[:, : self.output_channels]
            elif prepared.shape[1] < self.output_channels:
                repeat_count = self.output_channels - prepared.shape[1]
                pad = np.repeat(prepared[:, -1:], repeat_count, axis=1)
                prepared = np.concatenate((prepared, pad), axis=1)

        prepared = np.ascontiguousarray(prepared.astype(np.float32, copy=False))
        self.karaoke_audio = prepared

        if prepared.shape[0] <= 0:
            self.karaoke_position = 0
            self.karaoke_playing = False
            self.karaoke_finished = False
            return

        if keep_progress and previous_progress > 0.0:
            effective_now = int(
                np.clip(round(previous_progress * prepared.shape[0]), 0, prepared.shape[0])
            )
            if self.karaoke_position > sync_frames:
                self.karaoke_position = effective_now + sync_frames
            else:
                self.karaoke_position = effective_now
        else:
            self.karaoke_position = 0
        timeline_total = prepared.shape[0] + sync_frames
        if self.karaoke_position >= timeline_total:
            self.karaoke_position = 0
        self.karaoke_finished = False

    def set_karaoke_source(self, audio: np.ndarray, samplerate: int) -> None:
        """MR 오디오 소스를 설정하고 스트림 조건에 맞는 버퍼를 준비"""
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]
        if audio.ndim != 2:
            raise RuntimeError("Invalid karaoke audio format.")
        normalized = np.ascontiguousarray(audio.astype(np.float32, copy=False))
        with self.lock:
            self.karaoke_source_audio = normalized
            self.karaoke_source_sr = int(samplerate)
            self.karaoke_playing = False
            self.karaoke_position = 0
            self.karaoke_finished = False
            self._prepare_karaoke_audio_locked(keep_progress=False)

    def clear_karaoke_source(self) -> None:
        """MR 소스와 재생 상태를 모두 초기화"""
        with self.lock:
            self.karaoke_source_audio = None
            self.karaoke_source_sr = 0
            self.karaoke_audio = None
            self.karaoke_position = 0
            self.karaoke_playing = False
            self.karaoke_finished = False

    def set_karaoke_gain(self, gain: float) -> None:
        """MR 트랙 출력 볼륨 배율(0.0~2.0)을 설정"""
        with self.lock:
            self.karaoke_gain = float(np.clip(gain, 0.0, 2.0))

    def set_recording_sync_offset_ms(self, offset_ms: int) -> None:
        """녹음 파일 싱크 보정을 위한 수동 오프셋(ms)을 설정"""
        with self.lock:
            self.recording_sync_offset_ms = int(np.clip(offset_ms, -500, 500))

    def play_karaoke(self, restart: bool = False) -> bool:
        """MR 재생 시작(또는 재개). 준비된 트랙이 없으면 False"""
        with self.lock:
            if self.karaoke_audio is None or self.karaoke_audio.shape[0] <= 0:
                return False
            timeline_total = self.karaoke_audio.shape[0] + int(
                max(0, self.karaoke_sync_frames)
            )
            if restart or self.karaoke_position >= timeline_total:
                self.karaoke_position = 0
            self.karaoke_playing = True
            self.karaoke_finished = False
            return True

    def pause_karaoke(self) -> None:
        """MR 재생을 일시정지"""
        with self.lock:
            self.karaoke_playing = False

    def stop_karaoke(self) -> None:
        """MR 재생을 정지하고 위치를 처음으로 되돌림"""
        with self.lock:
            self.karaoke_playing = False
            self.karaoke_position = 0
            self.karaoke_finished = False

    def is_karaoke_playing(self) -> bool:
        """MR 재생 중 여부를 반환"""
        with self.lock:
            return bool(self.karaoke_playing)

    def get_karaoke_status(self) -> Tuple[bool, float, float, bool]:
        """(준비됨, 현재초, 전체초, 종료됨) 상태를 반환"""
        with self.lock:
            prepared = self.karaoke_audio is not None and self.karaoke_audio.shape[0] > 0
            if not prepared or self.sample_rate <= 0:
                return False, 0.0, 0.0, bool(self.karaoke_finished)
            total = self.karaoke_audio.shape[0] / float(self.sample_rate)
            effective_frame = int(
                np.clip(
                    self.karaoke_position - int(max(0, self.karaoke_sync_frames)),
                    0,
                    self.karaoke_audio.shape[0],
                )
            )
            current = effective_frame / float(self.sample_rate)
            return True, current, total, bool(self.karaoke_finished)

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

    def _build_wasapi_exclusive_settings(self) -> Optional[Tuple[Any, Any]]:
        """Windows WASAPI Exclusive 모드 사용 가능 시 extra_settings를 생성"""
        if (
            sys.platform != "win32"
            or not hasattr(sd, "WasapiSettings")
            or self.input_device is None
            or self.output_device is None
        ):
            return None

        try:
            in_dev = sd.query_devices(self.input_device)
            out_dev = sd.query_devices(self.output_device)
            in_api_name = str(sd.query_hostapis(int(in_dev["hostapi"]))["name"]).upper()
            out_api_name = str(
                sd.query_hostapis(int(out_dev["hostapi"]))["name"]
            ).upper()
            if "WASAPI" not in in_api_name or "WASAPI" not in out_api_name:
                return None
            return (sd.WasapiSettings(exclusive=True), sd.WasapiSettings(exclusive=True))
        except Exception:
            return None

    def _get_low_latency_hint(self) -> Tuple[float, float]:
        """장치 low latency 값을 바탕으로 요청 지연 힌트를 계산"""
        if self.input_device is None or self.output_device is None:
            return (0.01, 0.01)

        try:
            in_dev = sd.query_devices(self.input_device)
            out_dev = sd.query_devices(self.output_device)
            in_low = float(in_dev.get("default_low_input_latency") or 0.0)
            out_low = float(out_dev.get("default_low_output_latency") or 0.0)
        except Exception:
            return (0.01, 0.01)

        # 과도한 초저지연값은 노이즈/언더런을 유발할 수 있어 완만하게 제한한다.
        in_latency = min(max(in_low if in_low > 0 else 0.01, 0.006), 0.02)
        out_latency = min(max(out_low if out_low > 0 else 0.01, 0.006), 0.02)
        return (in_latency, out_latency)

    @staticmethod
    def _normalize_stream_status(status: sd.CallbackFlags) -> str:
        """콜백 상태 문자열에서 정보성 플래그를 제거해 UI용으로 정규화."""
        text = str(status).strip()
        if not text:
            return ""
        parts = [part.strip() for part in text.split(",") if part.strip()]
        filtered = [part for part in parts if part.lower() != "priming output"]
        return ", ".join(filtered)

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
        self.last_stream_status_at = 0.0
        self.latest_output = np.zeros(self.block_size, dtype=np.float32)
        with self.lock:
            self._prepare_karaoke_audio_locked(keep_progress=True)

        # 저지연 우선으로 시도하고, 장치가 지원하지 않으면 점진적으로 완화한다.
        latency_hint = self._get_low_latency_hint()
        wasapi_exclusive = self._build_wasapi_exclusive_settings()
        stream_attempts: List[Tuple[int, Any, Optional[Tuple[Any, Any]], str]] = []

        # 기본은 공유 모드의 안정 프로파일을 우선 사용한다.
        stream_attempts.extend(
            [
                (256, "low", None, "shared-low"),
                (128, "low", None, "shared-low"),
                (256, latency_hint, None, "shared-balanced"),
                (512, "low", None, "shared-fallback"),
                (1024, "high", None, "safe-fallback"),
            ]
        )
        # 장치가 잘 받는 경우에만 Exclusive를 마지막 후보로 시도한다.
        if wasapi_exclusive is not None:
            stream_attempts.extend(
                [
                    (256, "low", wasapi_exclusive, "wasapi-exclusive"),
                ]
            )
        errors: List[str] = []
        opened_stream: Optional[sd.Stream] = None
        selected_block_size = self.block_size
        selected_profile = self.stream_profile

        for (
            candidate_block_size,
            candidate_latency,
            candidate_extra_settings,
            candidate_profile,
        ) in stream_attempts:
            try:
                stream_kwargs: Dict[str, Any] = {
                    "device": (self.input_device, self.output_device),
                    "channels": (self.input_channels, self.output_channels),
                    "samplerate": self.sample_rate,
                    "blocksize": candidate_block_size,
                    "latency": candidate_latency,
                    "dtype": "float32",
                    "clip_off": True,
                    "dither_off": True,
                    "prime_output_buffers_using_stream_callback": True,
                    "callback": self._audio_callback,
                }
                if candidate_extra_settings is not None:
                    stream_kwargs["extra_settings"] = candidate_extra_settings

                opened_stream = sd.Stream(**stream_kwargs)
                opened_stream.start()
                if not opened_stream.active:
                    raise RuntimeError("stream did not become active")
                selected_block_size = candidate_block_size
                selected_profile = candidate_profile
                break
            except Exception as exc:
                errors.append(
                    "block="
                    f"{candidate_block_size}, latency={candidate_latency}, "
                    f"profile={candidate_profile}: {exc}"
                )
                if opened_stream is not None:
                    try:
                        opened_stream.close()
                    except Exception:
                        pass
                    opened_stream = None

        if opened_stream is None:
            error_text = "; ".join(errors) if errors else "unknown error"
            raise RuntimeError(f"Failed to open audio stream ({error_text})")

        self.block_size = selected_block_size
        self.stream_profile = selected_profile
        self._update_karaoke_sync_frames(getattr(opened_stream, "latency", None))
        self.latest_output = np.zeros(self.block_size, dtype=np.float32)
        self.stream = opened_stream
        return self.sample_rate

    def stop(self) -> None:
        """오디오 스트림을 안전하게 중지하고 녹음 상태를 정리"""
        with self.lock:
            stream = self.stream
            self.stream = None
            self.is_recording = False
            self.recorded_chunks = []
            self.recorded_mic_chunks = []
            self.recorded_karaoke_chunks = []
            self.recording_alignment_frames = 0
            self.active_record_output_mode = self.record_output_mode
            self.karaoke_playing = False
            self.karaoke_position = 0
            self.karaoke_finished = False
            self.last_stream_status = ""
            self.last_stream_status_at = 0.0

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
            karaoke_gain = float(self.karaoke_gain)
            karaoke_chunk: Optional[np.ndarray] = None
            if (
                self.karaoke_playing
                and self.karaoke_audio is not None
                and self.karaoke_audio.shape[0] > 0
            ):
                total = self.karaoke_audio.shape[0]
                delay = int(max(0, self.karaoke_sync_frames))
                timeline_total = total + delay
                start = int(np.clip(self.karaoke_position, 0, timeline_total))
                end = min(start + frames, timeline_total)
                karaoke_chunk = np.zeros(
                    (frames, self.karaoke_audio.shape[1]), dtype=np.float32
                )
                src_start = start - delay
                src_end = end - delay
                valid_start = max(src_start, 0)
                valid_end = min(src_end, total)
                if valid_end > valid_start:
                    dst_offset = valid_start - src_start
                    length = valid_end - valid_start
                    karaoke_chunk[dst_offset : dst_offset + length] = self.karaoke_audio[
                        valid_start:valid_end
                    ]
                self.karaoke_position = end
                if end >= timeline_total:
                    self.karaoke_playing = False
                    self.karaoke_finished = True
        status_text = self._normalize_stream_status(status)
        status_now = time.monotonic()

        if indata.size == 0:
            mono = np.zeros(frames, dtype=np.float32)
        else:
            mono = indata[:, 0].astype(np.float32, copy=True)

        processed = self._process_block(mono, params)
        mic_mix = np.zeros((frames, outdata.shape[1]), dtype=np.float32)
        mic_mix[:, 0] = processed
        if outdata.shape[1] > 1:
            mic_mix[:, 1:] = processed[:, np.newaxis]

        karaoke_mix = np.zeros((frames, outdata.shape[1]), dtype=np.float32)
        mixed_for_output = np.array(mic_mix, copy=True)
        if outdata.shape[1] > 1:
            mixed_for_output[:, 1:] = processed[:, np.newaxis]
        if karaoke_chunk is not None and karaoke_chunk.size > 0:
            rows = karaoke_chunk.shape[0]
            cols = min(karaoke_chunk.shape[1], karaoke_mix.shape[1])
            karaoke_mix[:rows, :cols] += karaoke_chunk[:, :cols] * karaoke_gain
            if cols == 1 and karaoke_mix.shape[1] > 1:
                karaoke_mix[:rows, 1:] += karaoke_chunk[:, :1] * karaoke_gain
            mixed_for_output[:rows, :cols] += karaoke_chunk[:, :cols] * karaoke_gain
            if cols == 1 and mixed_for_output.shape[1] > 1:
                mixed_for_output[:rows, 1:] += karaoke_chunk[:, :1] * karaoke_gain
        np.clip(karaoke_mix, -1.0, 1.0, out=karaoke_mix)
        np.clip(mixed_for_output, -1.0, 1.0, out=mixed_for_output)

        # 녹음 중 무음 모드에서는 마이크 모니터링만 끄고 MR은 계속 출력.
        mute_output = recording_enabled and record_output_mode == "mute_while_recording"
        outdata[:] = karaoke_mix if mute_output else mixed_for_output

        with self.lock:
            if status_text:
                self.last_stream_status = status_text
                self.last_stream_status_at = status_now
            elif (
                self.last_stream_status
                and (status_now - float(self.last_stream_status_at)) >= 1.5
            ):
                # 경고가 더 이상 없으면 상태 표시는 자동으로 정리한다.
                self.last_stream_status = ""
            self.latest_output = processed
            if recording_enabled:
                # 녹음에는 마이크 이펙트 + MR이 합쳐진 최종 출력 신호를 저장한다.
                self.recorded_chunks.append(mixed_for_output.copy())
                self.recorded_mic_chunks.append(mic_mix.copy())
                self.recorded_karaoke_chunks.append(karaoke_mix.copy())

    def _process_block(self, samples: np.ndarray, params: EffectParams) -> np.ndarray:
        """한 블록의 오디오 샘플에 이펙트 체인을 순서대로 적용"""
        x = np.array(samples, dtype=np.float32, copy=True)
        if params.input_gain != 1.0:
            np.multiply(x, params.input_gain, out=x)

        if params.distortion_drive > 1.001:
            np.multiply(x, params.distortion_drive, out=x)
            np.tanh(x, out=x)

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
            np.multiply(x, params.output_gain, out=x)

        # 최종 출력은 [-1.0, 1.0] 범위로 제한해 클리핑 왜곡을 방지한다.
        np.clip(x, -1.0, 1.0, out=x)
        return x

    def _apply_reverb(self, samples: np.ndarray, wet: float) -> np.ndarray:
        """짧은 멀티탭 버퍼를 이용해 간단한 리버브 효과를 적용"""
        wet = float(np.clip(wet, 0.0, 1.0))
        feedback = 0.35
        dry_mix = 1.0 - wet
        max_samples = self.max_reverb_samples

        for i in range(samples.size):
            dry = float(samples[i])
            reverb_sum = 0.0
            for tap, gain in zip(self.reverb_taps, self.reverb_tap_gains):
                read_idx = (self.reverb_idx - tap) % max_samples
                reverb_sum += float(self.reverb_buffer[read_idx]) * float(gain)

            self.reverb_buffer[self.reverb_idx] = dry + (reverb_sum * feedback)
            self.reverb_idx = (self.reverb_idx + 1) % max_samples
            samples[i] = (dry * dry_mix) + (reverb_sum * wet)

        return samples

    def _apply_delay(
        self, samples: np.ndarray, wet: float, delay_ms: float, feedback: float
    ) -> np.ndarray:
        """딜레이 시간/피드백 파라미터 기반의 딜레이 효과를 적용"""
        wet = float(np.clip(wet, 0.0, 1.0))
        dry_mix = 1.0 - wet
        feedback = float(np.clip(feedback, 0.0, 0.95))
        delay_samples = int(delay_ms * self.sample_rate / 1000.0)
        delay_samples = max(1, min(delay_samples, self.max_delay_samples - 1))
        max_samples = self.max_delay_samples

        for i in range(samples.size):
            dry = float(samples[i])
            read_idx = (self.delay_idx - delay_samples) % max_samples
            delayed = float(self.delay_buffer[read_idx])
            samples[i] = (dry * dry_mix) + (delayed * wet)
            self.delay_buffer[self.delay_idx] = dry + (delayed * feedback)
            self.delay_idx = (self.delay_idx + 1) % max_samples

        return samples

    def _apply_echo(
        self, samples: np.ndarray, wet: float, echo_ms: float, feedback: float
    ) -> np.ndarray:
        """긴 지연 기반의 에코 효과를 적용"""
        wet = float(np.clip(wet, 0.0, 1.0))
        dry_mix = 1.0 - wet
        feedback = float(np.clip(feedback, 0.0, 0.95))
        echo_samples = int(echo_ms * self.sample_rate / 1000.0)
        echo_samples = max(1, min(echo_samples, self.max_echo_samples - 1))
        max_samples = self.max_echo_samples

        for i in range(samples.size):
            dry = float(samples[i])
            read_idx = (self.echo_idx - echo_samples) % max_samples
            echoed = float(self.echo_buffer[read_idx])
            samples[i] = (dry * dry_mix) + (echoed * wet)
            self.echo_buffer[self.echo_idx] = dry + (echoed * feedback)
            self.echo_idx = (self.echo_idx + 1) % max_samples

        return samples

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
            self.recorded_mic_chunks = []
            self.recorded_karaoke_chunks = []
            manual_frames = int(
                round((self.recording_sync_offset_ms / 1000.0) * float(self.sample_rate))
            )
            self.recording_alignment_frames = int(self.input_latency_frames + manual_frames)
            self.is_recording = True

    def stop_recording(self) -> Tuple[np.ndarray, int]:
        """녹음을 종료하고 누적된 오디오 데이터와 샘플레이트를 반환"""
        with self.lock:
            self.is_recording = False
            self.active_record_output_mode = self.record_output_mode
            chunks = self.recorded_chunks
            self.recorded_chunks = []
            mic_chunks = self.recorded_mic_chunks
            karaoke_chunks = self.recorded_karaoke_chunks
            self.recorded_mic_chunks = []
            self.recorded_karaoke_chunks = []
            align_frames = int(max(0, self.recording_alignment_frames))
            self.recording_alignment_frames = 0
            rate = self.sample_rate

        if not chunks:
            return np.array([], dtype=np.float32), rate

        # 녹음 파일에서는 자동 지연 + 수동 보정(ms)을 반영해 MR/보컬 싱크를 정렬한다.
        if (
            mic_chunks
            and karaoke_chunks
            and len(mic_chunks) == len(karaoke_chunks)
        ):
            mic = np.concatenate(mic_chunks, axis=0).astype(np.float32, copy=False)
            karaoke = np.concatenate(karaoke_chunks, axis=0).astype(np.float32, copy=False)
            if not np.any(np.abs(karaoke) > 1e-7):
                return np.concatenate(chunks).astype(np.float32), rate
            if mic.ndim == 1:
                mic = mic[:, np.newaxis]
            if karaoke.ndim == 1:
                karaoke = karaoke[:, np.newaxis]
            channels = max(mic.shape[1], karaoke.shape[1])
            if mic.shape[1] < channels:
                mic = np.repeat(mic[:, :1], channels, axis=1)
            if karaoke.shape[1] < channels:
                karaoke = np.repeat(karaoke[:, :1], channels, axis=1)

            if align_frames >= 0:
                mic_start = 0
                karaoke_start = align_frames
            else:
                mic_start = -align_frames
                karaoke_start = 0

            total = max(
                mic_start + mic.shape[0],
                karaoke_start + karaoke.shape[0],
            )
            mixed = np.zeros((total, channels), dtype=np.float32)
            mixed[mic_start : mic_start + mic.shape[0], : mic.shape[1]] += mic
            mixed[
                karaoke_start : karaoke_start + karaoke.shape[0], : karaoke.shape[1]
            ] += karaoke
            np.clip(mixed, -1.0, 1.0, out=mixed)
            return mixed, rate

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
            "group_karaoke": "노래방",
            "tab_effects": "이펙트",
            "tab_studio": "스튜디오",
            "tab_karaoke": "노래방",
            "tab_waveforms": "파형 보기",
            "tab_recording": "녹음/재생",
            "label_input_mic": "입력 마이크",
            "label_output_speaker": "출력 스피커",
            "label_language": "언어",
            "label_theme": "테마",
            "label_process_end_mode": "처리 종료 모드",
            "tooltip_help": "사용법 보기",
            "dialog_help_title": "온단비 VE 스튜디오 사용법",
            "msg_help_content": (
                "1) 오디오 장치에서 입력/출력을 선택하고 [처리 시작]을 누르세요.\n"
                "2) 이펙트 탭에서 게인/리버브/딜레이 등을 원하는 만큼 조절하세요.\n"
                "3) 스튜디오 탭에서 [시작]-[정지]로 녹음하고 바로 재생해 확인하세요.\n"
                "4) 노래방 탭에서 MR/노래 파일을 불러와 [재생]/[정지]로 연습하세요.\n"
                "5) 싱크가 어긋나면 [녹음 싱크 보정] 값을 먼저 110ms 근처에서 조절하세요."
            ),
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
            "stream_status_output_underflow": "출력 버퍼 지연",
            "stream_status_input_underflow": "입력 버퍼 지연",
            "stream_status_output_overflow": "출력 버퍼 과부하",
            "stream_status_input_overflow": "입력 버퍼 과부하",
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
            "msg_m4a_dep_missing": "M4A/MP3 등 일부 포맷 처리를 위해 `imageio-ffmpeg` 패키지가 필요합니다.",
            "msg_m4a_convert_failed": "M4A 변환/로딩에 실패했습니다: {reason}",
            "msg_audio_decode_failed": "오디오 파일 로딩에 실패했습니다: {reason}",
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
            "label_karaoke_track": "MR/노래 파일",
            "btn_karaoke_browse": "파일 불러오기",
            "label_karaoke_gain": "MR 볼륨",
            "label_record_sync_offset": "녹음 싱크 보정",
            "btn_karaoke_play": "재생",
            "btn_karaoke_pause": "일시정지",
            "btn_karaoke_stop": "정지",
            "status_karaoke_idle": "노래방 트랙이 선택되지 않았습니다.",
            "status_karaoke_loaded": "트랙 로드됨: {path} ({duration:.2f}초)",
            "status_karaoke_playing": "노래방 재생 중: {current:.2f}초 / {total:.2f}초",
            "status_karaoke_paused": "노래방 일시정지: {current:.2f}초 / {total:.2f}초",
            "status_karaoke_stopped": "노래방 정지(0초): {path}",
            "status_karaoke_finished": "노래방 재생 완료: {path}",
            "dialog_karaoke_file_error": "노래방 파일 오류",
            "msg_karaoke_select_file": "먼저 MR/노래 파일을 불러오세요.",
            "filter_audio_load": "오디오 파일 (*.wav *.mp3 *.m4a *.flac *.ogg *.opus *.aac *.wma);;모든 파일 (*.*)",
        },
        "en": {
            "window_title": "Ondanbi VE Studio",
            "group_audio_devices": "Audio Devices",
            "group_effects_gain": "Effects and Gain",
            "group_waveforms": "Waveforms",
            "group_recording": "Recording",
            "group_karaoke": "Karaoke",
            "tab_effects": "Effects",
            "tab_studio": "Studio",
            "tab_karaoke": "Karaoke",
            "tab_waveforms": "Waveforms",
            "tab_recording": "Record/Playback",
            "label_input_mic": "Input Mic",
            "label_output_speaker": "Output Speaker",
            "label_language": "Language",
            "label_theme": "Theme",
            "label_process_end_mode": "Process End Mode",
            "tooltip_help": "Open quick guide",
            "dialog_help_title": "Ondanbi VE Studio Guide",
            "msg_help_content": (
                "1) Select input/output devices, then click [Start Processing].\n"
                "2) Adjust gain, reverb, delay, and other effects in the Effects tab.\n"
                "3) In Studio tab, record with [Start]/[Stop] and play back immediately.\n"
                "4) In Karaoke tab, load an MR/song track and control with [Play]/[Stop].\n"
                "5) If recorded vocal timing is off, tune [Record Sync Offset] near 110ms."
            ),
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
            "stream_status_output_underflow": "Output underflow",
            "stream_status_input_underflow": "Input underflow",
            "stream_status_output_overflow": "Output overflow",
            "stream_status_input_overflow": "Input overflow",
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
            "msg_m4a_dep_missing": "`imageio-ffmpeg` is required for M4A/MP3 and some additional formats.",
            "msg_m4a_convert_failed": "M4A conversion/loading failed: {reason}",
            "msg_audio_decode_failed": "Failed to load audio file: {reason}",
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
            "label_karaoke_track": "MR/Song File",
            "btn_karaoke_browse": "Load File",
            "label_karaoke_gain": "MR Volume",
            "label_record_sync_offset": "Record Sync Offset",
            "btn_karaoke_play": "Play",
            "btn_karaoke_pause": "Pause",
            "btn_karaoke_stop": "Stop",
            "status_karaoke_idle": "No karaoke track selected.",
            "status_karaoke_loaded": "Track loaded: {path} ({duration:.2f}s)",
            "status_karaoke_playing": "Karaoke playing: {current:.2f}s / {total:.2f}s",
            "status_karaoke_paused": "Karaoke paused: {current:.2f}s / {total:.2f}s",
            "status_karaoke_stopped": "Karaoke stopped (0s): {path}",
            "status_karaoke_finished": "Karaoke finished: {path}",
            "dialog_karaoke_file_error": "Karaoke File Error",
            "msg_karaoke_select_file": "Load an MR/song file first.",
            "filter_audio_load": "Audio files (*.wav *.mp3 *.m4a *.flac *.ogg *.opus *.aac *.wma);;All files (*.*)",
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
        self.current_theme = self._detect_initial_theme()
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
        self.karaoke_track_path: Optional[Path] = None
        self.karaoke_ui_signature: Optional[Tuple[str, float, float, bool, bool, str]] = None
        self.live_wave_visible = False
        self.last_cursor_frame_synced = -1
        self.stream_button_running_state: Optional[bool] = None
        self.playback_button_signature: Optional[Tuple[bool, bool, bool, str]] = None
        self.stream_expected_running = False
        self.last_system_theme = self.current_theme

        self._build_ui()
        self._apply_theme()
        self._apply_language()
        self._load_devices()
        self._sync_all_params()
        self._connect_system_theme_sync()

        self.wave_timer = QTimer(self)
        # 파형 갱신 빈도를 약간 낮춰 UI 부하와 GIL 경합을 줄인다.
        self.wave_timer.setInterval(80)
        self.wave_timer.timeout.connect(self._update_live_waveform)
        self.wave_timer.start()

        self.system_theme_timer = QTimer(self)
        self.system_theme_timer.setInterval(2000)
        self.system_theme_timer.timeout.connect(self._poll_system_theme)
        self.system_theme_timer.start()

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

    def _format_stream_status_for_ui(self, raw_status: str) -> str:
        """오디오 상태 플래그를 현재 언어의 짧은 설명으로 변환."""
        status_text = raw_status.strip()
        if not status_text:
            return ""
        lower = status_text.lower()
        mapping = (
            ("output underflow", "stream_status_output_underflow"),
            ("input underflow", "stream_status_input_underflow"),
            ("output overflow", "stream_status_output_overflow"),
            ("input overflow", "stream_status_input_overflow"),
        )
        labels: List[str] = []
        for token, key in mapping:
            if token in lower:
                labels.append(self._t(key))
        if labels:
            # 동일 플래그 중복 표시는 제거한다.
            unique_labels = list(dict.fromkeys(labels))
            return ", ".join(unique_labels)
        return status_text

    def _detect_initial_theme(self) -> str:
        """앱 시작 시 시스템 테마를 감지해 초기 테마(light/dark)를 반환."""
        # 1) Qt 스타일 힌트 우선 사용 (지원 시 OS 테마를 직접 반영)
        try:
            app = QApplication.instance()
            if app is not None:
                hints = app.styleHints()
                if hasattr(hints, "colorScheme"):
                    color_scheme = hints.colorScheme()
                    if color_scheme == Qt.ColorScheme.Dark:
                        return "dark"
                    if color_scheme == Qt.ColorScheme.Light:
                        return "light"
        except Exception:
            pass

        # 2) Windows 레지스트리 fallback (AppsUseLightTheme: 0=dark, 1=light)
        if sys.platform == "win32":
            try:
                import winreg

                key_path = (
                    r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
                )
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                    value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
                    return "light" if int(value) == 1 else "dark"
            except Exception:
                pass

        # 3) 감지 실패 시 안전 기본값
        return "light"

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

    def _show_help_dialog(self) -> None:
        """상단 도움말 버튼 클릭 시 간단 사용법 팝업을 표시."""
        QMessageBox.information(
            self,
            self._t("dialog_help_title"),
            self._t("msg_help_content"),
        )

    def _connect_system_theme_sync(self) -> None:
        """OS 테마 변경 시그널이 있으면 연결해 즉시 반영한다."""
        try:
            app = QApplication.instance()
            if app is None:
                return
            hints = app.styleHints()
            signal = getattr(hints, "colorSchemeChanged", None)
            if signal is not None:
                signal.connect(self._on_system_theme_changed)
        except Exception:
            pass

    def _on_system_theme_changed(self, *_args) -> None:
        """Qt 시그널로 전달된 OS 테마 변경을 즉시 반영."""
        self._sync_theme_with_system(force=True)

    def _poll_system_theme(self) -> None:
        """시그널 미지원/누락 환경을 대비한 주기적 OS 테마 동기화."""
        self._sync_theme_with_system(force=False)

    def _sync_theme_with_system(self, force: bool = False) -> None:
        """현재 OS 테마와 앱 테마를 동기화한다."""
        detected_theme = self._detect_initial_theme()
        if (not force) and detected_theme == self.last_system_theme:
            return
        self.last_system_theme = detected_theme
        if self.current_theme != detected_theme:
            self.current_theme = detected_theme
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
    def _format_seconds_text(seconds: float) -> str:
        """초 단위를 mm:ss 형식 문자열로 변환"""
        seconds = max(0.0, float(seconds))
        total = int(round(seconds))
        minutes = total // 60
        remain = total % 60
        return f"{minutes:02d}:{remain:02d}"

    def _on_karaoke_gain_changed(self, value: int) -> None:
        """노래방 MR 볼륨 슬라이더 값을 엔진에 반영"""
        gain = float(value) / 100.0
        self.engine.set_karaoke_gain(gain)
        if not self.karaoke_gain_value.hasFocus():
            self.karaoke_gain_value.setText(f"{value}%")
        self._refresh_karaoke_controls(force=True)

    def _on_record_sync_offset_changed(self, value: int) -> None:
        """녹음 파일 싱크 보정(ms) 값을 엔진에 반영"""
        self.engine.set_recording_sync_offset_ms(int(value))
        if not self.record_sync_offset_value.hasFocus():
            self.record_sync_offset_value.setText(f"{int(value):+d} ms")

    def _apply_karaoke_gain_input(self) -> None:
        """MR 볼륨 입력 필드 텍스트를 슬라이더 값으로 반영"""
        parsed = self._parse_slider_input_value(
            text=self.karaoke_gain_value.text(),
            current_value=self.karaoke_gain_slider.value(),
            slider=self.karaoke_gain_slider,
        )
        self.karaoke_gain_slider.setValue(parsed)
        self.karaoke_gain_value.setText(f"{self.karaoke_gain_slider.value()}%")

    def _apply_record_sync_offset_input(self) -> None:
        """녹음 싱크 보정 입력 필드 텍스트를 슬라이더 값으로 반영"""
        parsed = self._parse_slider_input_value(
            text=self.record_sync_offset_value.text(),
            current_value=self.record_sync_offset_slider.value(),
            slider=self.record_sync_offset_slider,
        )
        self.record_sync_offset_slider.setValue(parsed)
        self.record_sync_offset_value.setText(
            f"{int(self.record_sync_offset_slider.value()):+d} ms"
        )

    def _load_karaoke_track(self, source_path: Path) -> None:
        """선택한 MR/노래 파일을 로드해 엔진 카라오케 소스로 등록"""
        resolved = source_path.expanduser().resolve()
        data, samplerate = self._load_audio_file(resolved)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        self.engine.set_karaoke_source(data, int(samplerate))
        self.karaoke_track_path = resolved
        self.karaoke_path_edit.setText(str(resolved))
        duration = (
            data.shape[0] / float(samplerate) if samplerate and data.shape[0] > 0 else 0.0
        )
        self.karaoke_status.setText(
            self._t("status_karaoke_loaded", path=resolved, duration=duration)
        )
        self._refresh_karaoke_controls(force=True)

    def _browse_karaoke_file(self) -> None:
        """노래방 탭에서 MR/노래 파일을 선택해 로드"""
        current = self.karaoke_path_edit.text().strip() or str(Path.cwd())
        target, _ = QFileDialog.getOpenFileName(
            self, self._t("btn_karaoke_browse"), current, self._t("filter_audio_load")
        )
        if not target:
            return
        try:
            self._load_karaoke_track(Path(target))
        except Exception as exc:
            QMessageBox.critical(self, self._t("dialog_karaoke_file_error"), str(exc))

    def _toggle_karaoke_playback(self) -> None:
        """노래방 재생/일시정지 토글"""
        path_text = self.karaoke_path_edit.text().strip()
        if not path_text:
            QMessageBox.information(
                self, self._t("dialog_karaoke_file_error"), self._t("msg_karaoke_select_file")
            )
            return

        if self.karaoke_track_path is None:
            candidate = Path(path_text).expanduser()
            if not candidate.exists():
                QMessageBox.information(
                    self,
                    self._t("dialog_karaoke_file_error"),
                    self._t("msg_karaoke_select_file"),
                )
                return
            try:
                self._load_karaoke_track(candidate)
            except Exception as exc:
                QMessageBox.critical(
                    self, self._t("dialog_karaoke_file_error"), str(exc)
                )
                return

        if not self.engine.is_running():
            self._start_stream()
            if not self.engine.is_running():
                return

        self._stop_playback_stream()
        if self.engine.is_karaoke_playing():
            self.engine.pause_karaoke()
        else:
            if not self.engine.play_karaoke():
                QMessageBox.information(
                    self,
                    self._t("dialog_karaoke_file_error"),
                    self._t("msg_karaoke_select_file"),
                )
                return
        self._refresh_karaoke_controls(force=True)

    def _stop_karaoke_playback(self) -> None:
        """노래방 재생을 정지하고 재생 위치를 0초로 되돌림"""
        self.engine.stop_karaoke()
        if self.karaoke_track_path is not None:
            self.karaoke_status.setText(
                self._t("status_karaoke_stopped", path=self.karaoke_track_path)
            )
        else:
            self.karaoke_status.setText(self._t("status_karaoke_idle"))
        self._refresh_karaoke_controls(force=True)

    def _refresh_karaoke_controls(self, force: bool = False) -> None:
        """노래방 탭 버튼 상태/문구와 상태 라벨을 현재 엔진 상태로 동기화"""
        prepared, current, total, finished = self.engine.get_karaoke_status()
        playing = self.engine.is_karaoke_playing()
        signature = (
            "ready" if prepared else "empty",
            round(current, 2),
            round(total, 2),
            bool(playing),
            bool(finished),
            self.current_language,
        )
        if (not force) and self.karaoke_ui_signature == signature:
            return
        self.karaoke_ui_signature = signature
        if not self.karaoke_gain_value.hasFocus():
            self.karaoke_gain_value.setText(f"{int(self.karaoke_gain_slider.value())}%")
        if not self.record_sync_offset_value.hasFocus():
            self.record_sync_offset_value.setText(
                f"{int(self.record_sync_offset_slider.value()):+d} ms"
            )

        self._set_icon_button(
            self.karaoke_stop_btn,
            QStyle.StandardPixmap.SP_MediaStop,
            self._t("btn_karaoke_stop"),
        )
        self.karaoke_play_btn.setEnabled(prepared)
        self.karaoke_stop_btn.setEnabled(prepared and (playing or current > 0.0))

        if playing:
            self._set_icon_button(
                self.karaoke_play_btn,
                QStyle.StandardPixmap.SP_MediaPause,
                self._t("btn_karaoke_pause"),
            )
            self.karaoke_status.setText(
                self._t("status_karaoke_playing", current=current, total=total)
            )
        else:
            self._set_icon_button(
                self.karaoke_play_btn,
                QStyle.StandardPixmap.SP_MediaPlay,
                self._t("btn_karaoke_play"),
            )
            if prepared and finished and self.karaoke_track_path is not None:
                self.karaoke_status.setText(
                    self._t("status_karaoke_finished", path=self.karaoke_track_path)
                )
            elif prepared:
                if current <= 0.0001 and self.karaoke_track_path is not None:
                    self.karaoke_status.setText(
                        self._t(
                            "status_karaoke_loaded",
                            path=self.karaoke_track_path,
                            duration=total,
                        )
                    )
                else:
                    self.karaoke_status.setText(
                        self._t("status_karaoke_paused", current=current, total=total)
                    )
            else:
                self.karaoke_status.setText(self._t("status_karaoke_idle"))

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
            "karaoke_track_path": self.karaoke_path_edit.text().strip(),
            "karaoke_gain": int(self.karaoke_gain_slider.value()),
            "record_sync_offset_ms": int(self.record_sync_offset_slider.value()),
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

        karaoke_gain = payload.get("karaoke_gain")
        if karaoke_gain is not None:
            self.karaoke_gain_slider.setValue(
                self._clamp_slider_value(self.karaoke_gain_slider, karaoke_gain)
            )

        record_sync_offset = payload.get("record_sync_offset_ms")
        if record_sync_offset is not None:
            self.record_sync_offset_slider.setValue(
                self._clamp_slider_value(
                    self.record_sync_offset_slider, record_sync_offset
                )
            )

        karaoke_track = payload.get("karaoke_track_path")
        if isinstance(karaoke_track, str) and karaoke_track.strip():
            self.karaoke_path_edit.setText(karaoke_track.strip())
            karaoke_path = Path(karaoke_track.strip()).expanduser()
            if karaoke_path.exists():
                try:
                    self._load_karaoke_track(karaoke_path)
                except Exception:
                    # 세팅 로드 중 파일 파싱 실패는 치명적이지 않게 무시한다.
                    pass

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

    def _set_karaoke_button_sizes(self) -> None:
        """노래방 재생 컨트롤 버튼 크기를 동일하게 고정."""
        for button in (self.karaoke_play_btn, self.karaoke_stop_btn):
            button.setMinimumWidth(128)
            button.setMaximumWidth(128)
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
            QPushButton#helpButton {{
                background: {colors["input_bg"]};
                color: {colors["text"]};
                border: 1px solid {colors["input_border"]};
                border-radius: 8px;
                min-width: 34px;
                max-width: 34px;
                min-height: 34px;
                max-height: 34px;
                padding: 0px;
                font-size: 13pt;
                font-weight: 700;
            }}
            QPushButton#helpButton:hover {{
                background: {colors["tab_hover"]};
            }}
            QPushButton#helpButton:pressed {{
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

        # 최상단 우측 빠른 도움말 버튼
        self.help_btn = QPushButton("?")
        self.help_btn.setObjectName("helpButton")
        self.help_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.help_btn.setFixedSize(34, 34)
        self.help_btn.clicked.connect(self._show_help_dialog)
        utility_row = QHBoxLayout()
        utility_row.setContentsMargins(0, 0, 0, 0)
        utility_row.addStretch(1)
        utility_row.addWidget(self.help_btn, 0, Qt.AlignmentFlag.AlignRight)
        root_layout.addLayout(utility_row)

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

        # 5) 노래방 탭 (MR 파일 + 마이크 이펙트 동시 출력)
        self.karaoke_tab = QWidget()
        karaoke_tab_layout = QVBoxLayout(self.karaoke_tab)
        karaoke_tab_layout.setContentsMargins(8, 8, 8, 8)
        karaoke_tab_layout.setSpacing(10)

        self.karaoke_group = QGroupBox()
        karaoke_layout = QGridLayout(self.karaoke_group)
        karaoke_layout.setHorizontalSpacing(10)
        karaoke_layout.setVerticalSpacing(8)

        self.karaoke_path_label = QLabel()
        self.karaoke_path_edit = QLineEdit()
        self.karaoke_path_edit.setReadOnly(True)
        self.karaoke_path_edit.setMinimumWidth(380)
        self.karaoke_path_edit.setMaximumWidth(430)
        self.karaoke_browse_btn = QPushButton()
        self.karaoke_browse_btn.clicked.connect(self._browse_karaoke_file)
        # 우측 컨트롤 폭을 통일해 정렬감을 유지한다.
        self.karaoke_browse_btn.setMinimumWidth(104)
        self.karaoke_browse_btn.setMaximumWidth(104)
        self.karaoke_browse_btn.setMinimumHeight(32)

        self.karaoke_gain_label = QLabel()
        self.karaoke_gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.karaoke_gain_slider.setRange(0, 200)
        self.karaoke_gain_slider.setValue(100)
        self.karaoke_gain_slider.setMinimumWidth(260)
        self.karaoke_gain_slider.setMaximumWidth(430)
        self.karaoke_gain_slider.valueChanged.connect(self._on_karaoke_gain_changed)
        self.karaoke_gain_value = QLineEdit("100%")
        self.karaoke_gain_value.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.karaoke_gain_value.setMinimumWidth(88)
        self.karaoke_gain_value.setMaximumWidth(104)
        self.karaoke_gain_value.editingFinished.connect(self._apply_karaoke_gain_input)

        self.record_sync_offset_label = QLabel()
        self.record_sync_offset_slider = QSlider(Qt.Orientation.Horizontal)
        self.record_sync_offset_slider.setRange(-300, 300)
        self.record_sync_offset_slider.setValue(110)
        self.record_sync_offset_slider.setMinimumWidth(260)
        self.record_sync_offset_slider.setMaximumWidth(430)
        self.record_sync_offset_slider.valueChanged.connect(
            self._on_record_sync_offset_changed
        )
        self.record_sync_offset_value = QLineEdit("+110 ms")
        self.record_sync_offset_value.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.record_sync_offset_value.setMinimumWidth(88)
        self.record_sync_offset_value.setMaximumWidth(104)
        self.record_sync_offset_value.editingFinished.connect(
            self._apply_record_sync_offset_input
        )

        self.karaoke_play_btn = QPushButton()
        self.karaoke_play_btn.clicked.connect(self._toggle_karaoke_playback)
        self.karaoke_stop_btn = QPushButton()
        self.karaoke_stop_btn.clicked.connect(self._stop_karaoke_playback)
        self.karaoke_stop_btn.setEnabled(False)
        self._set_karaoke_button_sizes()

        self.karaoke_status = QLabel()
        self.karaoke_status.setObjectName("statusPill")
        self.karaoke_status.setMaximumWidth(660)
        self.karaoke_status.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        # 노래방 컨트롤 블록을 중앙으로 모으기 위한 6열 레이아웃
        # 0: 라벨 / 1: 좌 여백 / 2: 메인 컨트롤 / 3: 간격 / 4: 우측 컨트롤 / 5: 우 여백
        karaoke_layout.addWidget(self.karaoke_path_label, 0, 0)
        karaoke_layout.addWidget(
            self.karaoke_path_edit, 0, 2, 1, 1, Qt.AlignmentFlag.AlignHCenter
        )
        karaoke_layout.addWidget(
            self.karaoke_browse_btn, 0, 4, 1, 1, Qt.AlignmentFlag.AlignHCenter
        )
        karaoke_layout.addWidget(self.karaoke_gain_label, 1, 0)
        karaoke_layout.addWidget(
            self.karaoke_gain_slider, 1, 2, 1, 1, Qt.AlignmentFlag.AlignHCenter
        )
        karaoke_layout.addWidget(
            self.karaoke_gain_value, 1, 4, 1, 1, Qt.AlignmentFlag.AlignHCenter
        )
        karaoke_layout.addWidget(self.record_sync_offset_label, 2, 0)
        karaoke_layout.addWidget(
            self.record_sync_offset_slider, 2, 2, 1, 1, Qt.AlignmentFlag.AlignHCenter
        )
        karaoke_layout.addWidget(
            self.record_sync_offset_value, 2, 4, 1, 1, Qt.AlignmentFlag.AlignHCenter
        )

        karaoke_action_row = QHBoxLayout()
        karaoke_action_row.setContentsMargins(0, 0, 0, 0)
        karaoke_action_row.setSpacing(10)
        karaoke_action_row.addStretch(1)
        karaoke_action_row.addWidget(self.karaoke_play_btn)
        karaoke_action_row.addWidget(self.karaoke_stop_btn)
        karaoke_action_row.addStretch(1)
        karaoke_layout.addLayout(karaoke_action_row, 3, 2, 1, 3)

        karaoke_layout.addWidget(
            self.karaoke_status, 4, 1, 1, 4, Qt.AlignmentFlag.AlignRight
        )

        karaoke_layout.setColumnMinimumWidth(0, 130)
        karaoke_layout.setColumnMinimumWidth(3, 14)
        karaoke_layout.setColumnStretch(1, 2)
        karaoke_layout.setColumnStretch(2, 0)
        karaoke_layout.setColumnStretch(4, 0)
        karaoke_layout.setColumnStretch(5, 3)

        karaoke_tab_layout.addWidget(self.karaoke_group)
        karaoke_tab_layout.addStretch(1)
        self.main_tabs.addTab(self.karaoke_tab, "")

        root_layout.addWidget(self.main_tabs, 1)

        self.setCentralWidget(root)

    def _apply_language(self) -> None:
        """현재 언어 기준으로 버튼, 라벨, 플롯 타이틀을 일괄 갱신"""
        self.setWindowTitle(self._t("window_title"))
        self.help_btn.setText("?")
        self.help_btn.setToolTip(self._t("tooltip_help"))
        self.help_btn.setStatusTip(self._t("tooltip_help"))
        self.device_group.setTitle(self._t("group_audio_devices"))
        self.presets_group.setTitle(self._t("group_presets_settings"))
        self.effects_group.setTitle(self._t("group_effects_gain"))
        self.plot_group.setTitle(self._t("group_waveforms"))
        self.rec_group.setTitle(self._t("group_recording"))
        self.karaoke_group.setTitle(self._t("group_karaoke"))
        self.main_tabs.setTabText(0, self._t("tab_effects"))
        self.main_tabs.setTabText(1, self._t("tab_studio"))
        self.main_tabs.setTabText(2, self._t("tab_karaoke"))

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

        self.karaoke_path_label.setText(self._t("label_karaoke_track"))
        self.karaoke_browse_btn.setText(self._t("btn_karaoke_browse"))
        self.karaoke_gain_label.setText(self._t("label_karaoke_gain"))
        if not self.karaoke_gain_value.hasFocus():
            self.karaoke_gain_value.setText(f"{int(self.karaoke_gain_slider.value())}%")
        self.record_sync_offset_label.setText(self._t("label_record_sync_offset"))
        if not self.record_sync_offset_value.hasFocus():
            self.record_sync_offset_value.setText(
                f"{int(self.record_sync_offset_slider.value()):+d} ms"
            )
        self._set_karaoke_button_sizes()

        for key, label in self.slider_title_labels.items():
            label.setText(self._t(key))

        if self.rec_stop_btn.isEnabled():
            self.record_status.setText(self._t("status_recording"))
        elif self.record_status.text().strip() == "":
            self.record_status.setText(self._t("status_no_recording"))

        if self.engine.is_running():
            status = self._format_stream_status_for_ui(self.engine.last_stream_status)
            if status:
                self.stream_label.setText(self._t("status_running_flag", status=status))
            else:
                self.stream_label.setText(
                    self._t("status_running_sr", samplerate=self.engine.sample_rate)
                )
        else:
            self.stream_label.setText(self._t("status_stopped"))

        self._refresh_playback_buttons(force=True)
        self._refresh_karaoke_controls(force=True)

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
        self.engine.set_karaoke_gain(self.karaoke_gain_slider.value() / 100.0)
        self.engine.set_recording_sync_offset_ms(int(self.record_sync_offset_slider.value()))

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
        except Exception as sf_exc:
            ffmpeg = self._get_ffmpeg_executable()
            with tempfile.TemporaryDirectory(prefix="ve_audio_decode_") as tmp_dir:
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
                except Exception as ff_exc:
                    reason = str(ff_exc).strip() or str(sf_exc).strip() or "unknown error"
                    raise RuntimeError(
                        self._t("msg_audio_decode_failed", reason=reason)
                    ) from ff_exc

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
        self._refresh_karaoke_controls(force=True)

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
        self._refresh_karaoke_controls(force=True)

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

        status = self._format_stream_status_for_ui(self.engine.last_stream_status)
        if self.engine.is_running():
            self._refresh_stream_toggle_button()
            if status:
                self.stream_label.setText(self._t("status_running_flag", status=status))
            self._refresh_karaoke_controls()
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
        self._refresh_karaoke_controls()

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
            if hasattr(self, "system_theme_timer"):
                self.system_theme_timer.stop()
            if hasattr(self, "wave_timer"):
                self.wave_timer.stop()
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
