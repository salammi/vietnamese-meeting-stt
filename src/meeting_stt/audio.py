from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, sosfilt


def pcm16_bytes_to_float32(data: bytes) -> np.ndarray:
    """Decode little-endian PCM16 and normalize it to [-1, 1]."""
    if len(data) % 2:
        data = data[:-1]
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def preprocess_far_field(
    audio: np.ndarray,
    sample_rate: int,
    high_pass_hz: float = 80,
    target_rms_db: float = -22,
    max_gain_db: float = 24,
) -> np.ndarray:
    """Remove DC/rumble and apply bounded RMS normalization for distant speech.

    The gain is bounded to avoid turning room noise into loud false speech. This is
    intentionally deterministic and lightweight; microphone-array beamforming or
    an RNNoise front end can be placed before this function in production.
    """
    signal = np.asarray(audio, dtype=np.float32)
    if signal.size == 0:
        return signal

    signal = signal - float(signal.mean())
    if 0 < high_pass_hz < sample_rate / 2:
        sos = butter(4, high_pass_hz, btype="highpass", fs=sample_rate, output="sos")
        signal = sosfilt(sos, signal).astype(np.float32)

    rms = float(np.sqrt(np.mean(np.square(signal), dtype=np.float64) + 1e-12))
    current_db = 20 * math.log10(max(rms, 1e-8))
    gain_db = min(max_gain_db, target_rms_db - current_db)
    signal *= 10 ** (gain_db / 20)

    peak = float(np.max(np.abs(signal)))
    if peak > 0.98:
        signal *= 0.98 / peak
    return signal.astype(np.float32)

