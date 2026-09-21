import numpy as np

from meeting_stt.audio import float32_to_pcm16, pcm16_bytes_to_float32, preprocess_far_field


def test_pcm_round_trip() -> None:
    original = np.array([-1.0, -0.5, 0.0, 0.5, 0.999], dtype=np.float32)
    decoded = pcm16_bytes_to_float32(float32_to_pcm16(original))
    assert np.allclose(original, decoded, atol=1 / 32768)


def test_preprocessing_is_finite_and_bounded() -> None:
    time = np.arange(16_000, dtype=np.float32) / 16_000
    audio = 0.003 * np.sin(2 * np.pi * 220 * time) + 0.05
    cleaned = preprocess_far_field(audio, 16_000)
    assert np.isfinite(cleaned).all()
    assert np.max(np.abs(cleaned)) <= 0.981
    assert abs(float(cleaned.mean())) < 0.01


def test_empty_audio() -> None:
    result = preprocess_far_field(np.array([], dtype=np.float32), 16_000)
    assert result.size == 0

