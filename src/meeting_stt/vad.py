from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import webrtcvad


@dataclass(slots=True)
class Utterance:
    pcm: bytes
    started_at_ms: int
    ended_at_ms: int


class UtteranceDetector:
    """WebRTC VAD state machine accepting 20 ms mono PCM16 frames."""

    frame_ms = 20

    def __init__(
        self,
        sample_rate: int = 16_000,
        aggressiveness: int = 2,
        min_utterance_ms: int = 400,
        end_silence_ms: int = 700,
        max_utterance_seconds: float = 28,
        pre_roll_ms: int = 300,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_bytes = sample_rate * self.frame_ms // 1000 * 2
        self.vad = webrtcvad.Vad(aggressiveness)
        self.min_frames = max(1, min_utterance_ms // self.frame_ms)
        self.silence_frames = max(1, end_silence_ms // self.frame_ms)
        self.max_frames = max(1, int(max_utterance_seconds * 1000 / self.frame_ms))
        self.pre_roll: deque[bytes] = deque(maxlen=max(1, pre_roll_ms // self.frame_ms))
        self.pending = bytearray()
        self.active: list[bytes] = []
        self.trailing_silence = 0
        self.frame_index = 0
        self.started_frame = 0

    def feed(self, chunk: bytes) -> list[Utterance]:
        self.pending.extend(chunk)
        ready: list[Utterance] = []
        while len(self.pending) >= self.frame_bytes:
            frame = bytes(self.pending[: self.frame_bytes])
            del self.pending[: self.frame_bytes]
            utterance = self._feed_frame(frame)
            if utterance:
                ready.append(utterance)
        return ready

    def flush(self) -> Utterance | None:
        if not self.active:
            return None
        return self._finish(force=True)

    def _feed_frame(self, frame: bytes) -> Utterance | None:
        is_speech = self.vad.is_speech(frame, self.sample_rate)
        self.frame_index += 1

        if not self.active:
            self.pre_roll.append(frame)
            if not is_speech:
                return None
            self.active = list(self.pre_roll)
            self.started_frame = self.frame_index - len(self.active)
            self.trailing_silence = 0
            return None

        self.active.append(frame)
        self.trailing_silence = 0 if is_speech else self.trailing_silence + 1
        if self.trailing_silence >= self.silence_frames:
            return self._finish(force=False)
        if len(self.active) >= self.max_frames:
            return self._finish(force=True)
        return None

    def _finish(self, force: bool) -> Utterance | None:
        frames = self.active if force else self.active[: -self.trailing_silence]
        end_frame = self.started_frame + len(frames)
        self.active = []
        self.trailing_silence = 0
        self.pre_roll.clear()
        if len(frames) < self.min_frames:
            return None
        return Utterance(
            pcm=b"".join(frames),
            started_at_ms=self.started_frame * self.frame_ms,
            ended_at_ms=end_frame * self.frame_ms,
        )
