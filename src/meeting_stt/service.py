from __future__ import annotations

import asyncio
import logging

import numpy as np

from .audio import preprocess_far_field
from .config import Settings
from .models import SpeakerClassifier, WhisperASR, select_candidate
from .schemas import TranscriptSegment
from .vad import Utterance

logger = logging.getLogger(__name__)


class TranscriptionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.vi_asr: WhisperASR | None = None
        self.mixed_asr: WhisperASR | None = None
        self.classifier: SpeakerClassifier | None = None
        self._inference_lock = asyncio.Lock()

    @property
    def loaded(self) -> bool:
        return self.vi_asr is not None

    def load(self) -> None:
        device = self.settings.resolved_device
        dtype = self.settings.resolved_dtype
        logger.info("Loading models on %s (%s)", device, dtype)
        self.vi_asr = WhisperASR(self.settings.vietnamese_asr_model_id, device, dtype)
        if self.settings.enable_mixed_asr and self.settings.mixed_asr_model_id:
            self.mixed_asr = WhisperASR(self.settings.mixed_asr_model_id, device, dtype)
        if self.settings.enable_classifier:
            try:
                self.classifier = SpeakerClassifier(
                    self.settings.classifier_model_id, device, dtype
                )
            except (OSError, RuntimeError, ValueError) as exc:
                # The upstream checkpoint declares a nonstandard architecture name.
                # ASR remains usable if a future Transformers version cannot map it.
                logger.exception("Speaker classifier could not be loaded: %s", exc)

    async def transcribe(self, utterance: Utterance) -> TranscriptSegment:
        async with self._inference_lock:
            return await asyncio.to_thread(self._transcribe_sync, utterance)

    def _transcribe_sync(self, utterance: Utterance) -> TranscriptSegment:
        if self.vi_asr is None:
            raise RuntimeError("Models have not been loaded")
        audio = np.frombuffer(utterance.pcm, dtype="<i2").astype(np.float32) / 32768.0
        audio = preprocess_far_field(
            audio,
            sample_rate=self.settings.sample_rate,
            high_pass_hz=self.settings.high_pass_hz,
            target_rms_db=self.settings.target_rms_db,
            max_gain_db=self.settings.max_gain_db,
        )
        vi_candidate = self.vi_asr.transcribe(audio, self.settings.sample_rate)
        mixed_candidate = (
            self.mixed_asr.transcribe(audio, self.settings.sample_rate)
            if self.mixed_asr
            else None
        )
        selected = select_candidate(vi_candidate, mixed_candidate)
        profile = (
            self.classifier.classify(audio, self.settings.sample_rate)
            if self.classifier
            else None
        )
        return TranscriptSegment(
            text=selected.text,
            language=selected.language,
            started_at_ms=utterance.started_at_ms,
            ended_at_ms=utterance.ended_at_ms,
            confidence=selected.confidence,
            model=selected.model_id,
            speaker_profile=profile,
        )
