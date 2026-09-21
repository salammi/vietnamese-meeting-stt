import struct

from meeting_stt.vad import UtteranceDetector


def frame(value: int = 0) -> bytes:
    return struct.pack("<320h", *([value] * 320))


def test_silence_does_not_create_utterance() -> None:
    detector = UtteranceDetector(sample_rate=16_000)
    assert detector.feed(frame() * 100) == []
    assert detector.flush() is None


def test_partial_network_chunks_are_buffered() -> None:
    detector = UtteranceDetector(sample_rate=16_000)
    one_frame = frame()
    assert detector.feed(one_frame[:137]) == []
    assert detector.feed(one_frame[137:]) == []

