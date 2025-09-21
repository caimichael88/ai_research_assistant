from __future__ import annotations
"""
Neural TTS Engine Composer

Composes a Text→Mel adapter (FastSpeech2/Tacotron2) with a Vocoder (HiFi‑GAN/WaveGlow)
behind the unified TTSEngine port.

Design goals:
- **Blocking API** (`synthesize_sync`) so routers can offload to a worker thread.
- Accept mel tensors as **numpy or torch**; leave conversion to the vocoder adapter.
- Optional **loudness normalization** hook (if you later implement one).
- Output **WAV** for minimal build; mp3/ogg can be added in adapters.
- Keep heavy deps out of this module; rely on adapters to carry them.
"""
from typing import Iterable, Optional
from dataclasses import dataclass
import time

try:
    from ..components.ports import (
        TTSEngine,
        SynthesisRequest,
        SynthesisResult,
        NeuralTextToMel,
        NeuralVocoder,
        LoudnessNormalizer,
        pcm16_to_wav,
    )
except ImportError:
    # Fallback for direct execution
    from components.ports import (
        TTSEngine,
        SynthesisRequest,
        SynthesisResult,
        NeuralTextToMel,
        NeuralVocoder,
        LoudnessNormalizer,
        pcm16_to_wav,
    )


@dataclass
class EngineDeps:
    t2m: NeuralTextToMel
    vocoder: NeuralVocoder
    loudness: Optional[LoudnessNormalizer] = None  # optional post‑processor


class NeuralTTSEngine(TTSEngine):
    """Compose Text→Mel + Vocoder behind a single TTSEngine entrypoint."""

    def __init__(self, t2m: NeuralTextToMel, vocoder: NeuralVocoder, loudness: Optional[LoudnessNormalizer] = None) -> None:
        self._deps = EngineDeps(t2m=t2m, vocoder=vocoder, loudness=loudness)

    # ---- TTSEngine port ----
    @property
    def id(self) -> str:
        return "neural"

    def warmup(self) -> None:
        # Delegate to adapters; safe to call multiple times
        self._deps.t2m.warmup()
        self._deps.vocoder.warmup()

    def synthesize_sync(self, req: SynthesisRequest) -> SynthesisResult:
        t0 = time.perf_counter()

        # 1) text → mel
        mel = self._deps.t2m.infer_mel(req.text, speed=req.speed)

        # 2) mel → PCM16 (adapter decides numpy/torch handling)
        pcm = self._deps.vocoder.infer_audio(mel, sample_rate=req.sample_rate)

        # 3) optional loudness normalization (bytes in, bytes out)
        if self._deps.loudness and req.normalize_loudness:
            try:
                pcm = self._deps.loudness.normalize(pcm, req.sample_rate)
            except Exception:
                # best‑effort; don't fail synthesis over loudness post‑fx
                pass

        # 4) containerize: minimal build supports WAV only
        media_type = "audio/wav"
        # If vocoder already returns WAV bytes, keep them; else wrap raw PCM16
        if not _looks_like_wav(pcm):
            audio_bytes = pcm16_to_wav(pcm, req.sample_rate)
        else:
            audio_bytes = pcm

        dt_ms = int((time.perf_counter() - t0) * 1000)
        return SynthesisResult(
            audio=audio_bytes,
            media_type=media_type,
            model_header=f"{self._deps.t2m.id}+{self._deps.vocoder.id}",
            latency_ms=dt_ms,
            cache="MISS",  # let outer service layer set HIT when applicable
        )

    # ---- Optional: streaming support (PCM16 chunks) ----
    def synthesize_stream_sync(self, req: SynthesisRequest) -> Iterable[bytes]:
        """Yield WAV bytes progressively (header + frames in chunks).
        Minimal implementation: synthesize fully then yield once.
        For true chunked streaming, implement chunked vocoder inference in the adapter.
        """
        result = self.synthesize_sync(req)
        yield result.audio


# --------------------- helpers ---------------------

def _looks_like_wav(b: bytes) -> bool:
    """Cheap check: 'RIFF' header & 'WAVE' format."""
    return isinstance(b, (bytes, bytearray)) and len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"WAVE"
