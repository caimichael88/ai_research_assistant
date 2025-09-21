# ── postfx/loudness.py
from __future__ import annotations
"""
Lightweight loudness post‑fx for TTS output.

Implements the LoudnessNormalizer port with a simple **RMS/peak** normalizer to roughly
hit a target loudness (in dBFS) and clamp peaks to avoid clipping.

Notes:
- This is a pragmatic stub (fast, stdlib+numpy). It is **not** a true LUFS (EBU R128)
  implementation. If you later add `pyloudnorm`, you can drop in a LUFS‑accurate
  normalizer behind the same interface.
- Input/Output: mono **PCM16** bytes (no WAV header), as produced by your vocoder.
"""
import math
from typing import Optional
import numpy as np  # type: ignore

try:
    from ..components.ports import LoudnessNormalizer
except ImportError:
    # Fallback for direct execution
    from components.ports import LoudnessNormalizer


class SimpleRMSNormalizer(LoudnessNormalizer):
    """Approximate loudness normalization by aligning RMS to a target dBFS,
    with a soft peak ceiling to avoid clipping.
    """
    def __init__(self, target_dbfs: float = -20.0, peak_ceiling: float = 0.98) -> None:
        self.target_dbfs = float(target_dbfs)
        self.peak_ceiling = float(peak_ceiling)

    def normalize(self, audio: bytes, sample_rate: int) -> bytes:
        # Note: sample_rate not used in RMS normalization but required by protocol
        if not audio:
            return audio
        # Interpret little‑endian int16 mono
        x = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
        if x.size == 0:
            return audio
        # Compute RMS in float domain (−32768..32767 → −1..1)
        x_norm = x / 32768.0
        rms = float(np.sqrt(np.mean(np.square(x_norm)))) + 1e-12
        curr_dbfs = 20.0 * math.log10(rms)
        gain_db = self.target_dbfs - curr_dbfs
        gain = 10.0 ** (gain_db / 20.0)
        y = x_norm * gain
        # Peak ceiling (soft clip)
        peak = float(np.max(np.abs(y))) + 1e-12
        if peak > self.peak_ceiling:
            y = y * (self.peak_ceiling / peak)
        # Back to int16
        y_int16 = np.clip(np.round(y * 32768.0), -32768, 32767).astype(np.int16)
        return y_int16.tobytes()


# Optional: LUFS (EBU R128) normalizer scaffold using pyloudnorm (if installed)
class LUFSDynamicNormalizer(LoudnessNormalizer):
    """Wrapper around pyloudnorm to target a LUFS value. Requires `pyloudnorm`.
    Falls back to SimpleRMSNormalizer if pyloudnorm is missing.
    """
    def __init__(self, target_lufs: float = -16.0) -> None:
        self.target_lufs = float(target_lufs)
        try:
            import pyloudnorm as pyln  # type: ignore
        except Exception:
            self._fallback: Optional[SimpleRMSNormalizer] = SimpleRMSNormalizer(target_dbfs=-20.0)
            self._pyln = None
        else:
            self._fallback = None
            self._pyln = pyln

    def normalize(self, audio: bytes, sample_rate: int) -> bytes:
        if getattr(self, "_pyln", None) is None:
            return self._fallback.normalize(audio, sample_rate)  # type: ignore
        import numpy as np  # type: ignore
        x = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        if x.size == 0:
            return audio
        meter = self._pyln.Meter(sample_rate)  # type: ignore
        loudness = meter.integrated_loudness(x)
        gain_db = self.target_lufs - loudness
        gain = 10.0 ** (gain_db / 20.0)
        y = x * gain
        y_int16 = np.clip(np.round(y * 32768.0), -32768, 32767).astype(np.int16)
        return y_int16.tobytes()


