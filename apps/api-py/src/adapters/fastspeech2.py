from __future__ import annotations
"""
FastSpeech 2 Adapter (Text → Mel)

This adapter implements the `NeuralTextToMel` port using either:
- **PyTorch** checkpoints (CPU by default; GPU if available), or
- **ONNX Runtime** sessions for faster CPU inference on Mac/servers.

It deliberately avoids importing heavy deps at module import time. All heavy
imports happen inside `_ensure_loaded()`.

Minimal usage (CPU on Mac):
    from adapters.fastspeech2 import FastSpeech2Adapter
    t2m = FastSpeech2Adapter(ckpt_path="/models/fastspeech2.onnx", backend="onnx", device="cpu")
    t2m.warmup()
    mel = t2m.infer_mel("Hello world")

Notes:
- This is a **scaffold**. You must plug in your actual tokenizer/G2P and model
  loading details according to the FS2 implementation you use (ESPnet, Coqui-TTS,
  PaddleSpeech, NVIDIA, etc.). Shapes & input names may differ.
- `speed` maps to FastSpeech2's duration control. Typical FS2 uses **pace** where
  `pace = 1/speed`. We handle that conversion below.
"""
from dataclasses import dataclass
from typing import Any, Optional
import os

try:
    from ..components.ports import NeuralTextToMel, MelSpectrogram, TTSSynthesisError
except ImportError:
    # Fallback for direct execution
    from components.ports import NeuralTextToMel, MelSpectrogram, TTSSynthesisError


# ------------------------ Config & Defaults ------------------------
@dataclass(frozen=True)
class FS2Config:
    n_mels: int = 80
    hop_length: int = 256
    sample_rate: int = 22050
    backend: str = "onnx"          # "onnx" | "torch"
    device: str = "cpu"            # "cpu" | "cuda"
    dtype: str = "float32"         # only for torch path
    # Tokenizer / vocab
    vocab_path: Optional[str] = None  # path to vocab / symbol table if needed


# ------------------------ Adapter ------------------------
class FastSpeech2Adapter(NeuralTextToMel):
    def __init__(
        self,
        ckpt_path: str,
        device: Optional[str] = None,
        voice_id: str = "en_female_1",
        backend: Optional[str] = None,
        config: Optional[FS2Config] = None,
    ) -> None:
        """
        :param ckpt_path: Path to FS2 checkpoint (ONNX .onnx or Torch .pt/.pth)
        :param device:    "cpu" (default on Mac) or "cuda"
        :param voice_id:  Logical voice identifier used by the service
        :param backend:   "onnx" or "torch". If None, inferred from file suffix
        :param config:    Optional FS2Config to override defaults
        """
        self._ckpt_path = ckpt_path
        self._voice_id = voice_id
        self._loaded = False

        if config is None:
            config = FS2Config()
        dev = device or os.getenv("DEVICE", config.device)
        bkd = backend or ("onnx" if ckpt_path.endswith(".onnx") else config.backend)

        # store resolved
        self._config = FS2Config(
            n_mels=config.n_mels,
            hop_length=config.hop_length,
            sample_rate=config.sample_rate,
            backend=bkd,
            device=dev,
            dtype=config.dtype,
            vocab_path=config.vocab_path,
        )

        # backends (lazy)
        self._ort_session = None
        self._torch_model = None
        self._symbols = None  # tokenizer symbols / vocab

    # ---- Port properties ----
    @property
    def id(self) -> str:
        return "fastspeech2"

    @property
    def voice_id(self) -> str:
        return self._voice_id

    # ---- Lifecycle ----
    def warmup(self) -> None:
        self._ensure_loaded()
        # Optional: run a tiny forward with a short text
        try:
            _ = self.infer_mel("warmup")
        except Exception:
            # don't hard-fail on warmup; the real request can surface the error
            pass

    # ---- Core ----
    def infer_mel(self, text: str, *, speed: float = 1.0) -> MelSpectrogram:
        self._ensure_loaded()
        if not text:
            raise TTSSynthesisError("empty text")

        # 1) normalize + tokenize
        norm_text = self._normalize(text)
        ids = self._text_to_sequence(norm_text)
        if len(ids) == 0:
            raise TTSSynthesisError("no valid tokens after normalization")

        # 2) map `speed` (1.0 = normal) to FS2 `pace` (1.0 = normal, >1 faster)
        # Common mapping: pace = 1 / speed (guard division by zero)
        pace = 1.0 / max(0.1, float(speed))

        # 3) run backend
        if self._config.backend == "onnx":
            mel = self._infer_onnx(ids, pace)
        else:
            mel = self._infer_torch(ids, pace)

        # 4) wrap result
        return MelSpectrogram(
            data=mel,
            sample_rate=self._config.sample_rate,
            hop_length=self._config.hop_length,
            n_mels=self._config.n_mels,
        )

    # ------------------------ Internal helpers ------------------------
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        # Load tokenizer / symbols if provided
        self._symbols = self._load_symbols(self._config.vocab_path)

        if self._config.backend == "onnx":
            try:
                import onnxruntime as ort  # type: ignore
            except Exception as e:
                raise TTSSynthesisError(f"onnxruntime not available: {e}")

            sess_opts = ort.SessionOptions()
            # perf knobs (tune as needed)
            sess_opts.intra_op_num_threads = max(1, os.cpu_count() or 1)
            # Set up execution providers based on device
            if self._config.device == "cpu":
                providers = ["CPUExecutionProvider"]
            elif self._config.device == "cuda":
                providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
            elif self._config.device == "mps":
                providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            else:
                # Fallback to CPU for unknown devices
                providers = ["CPUExecutionProvider"]
            try:
                self._ort_session = ort.InferenceSession(self._ckpt_path, sess_opts, providers=providers)
            except Exception as e:
                raise TTSSynthesisError(f"failed to load ONNX model: {e}")
        else:
            try:
                import torch  # type: ignore
            except Exception as e:
                raise TTSSynthesisError(f"PyTorch not available: {e}")

            # Handle device mapping for PyTorch loading
            if self._config.device in ("cpu", "mps"):
                map_location = "cpu"  # MPS models are typically saved as CPU tensors
            else:
                map_location = None  # Let PyTorch handle CUDA device mapping
            try:
                # Replace with the actual loading routine for your FS2 implementation
                ckpt = torch.load(self._ckpt_path, map_location=map_location)
                self._torch_model = self._build_torch_model_from_ckpt(ckpt, device=self._config.device)
                self._torch_model.eval()
            except Exception as e:
                raise TTSSynthesisError(f"failed to load Torch model: {e}")

        self._loaded = True

    # ---- Normalization / Tokenization (replace with your real logic) ----
    def _normalize(self, text: str) -> str:
        # Minimal cleaner; plug your real number expansion / punctuation handling here
        return " ".join(text.strip().split())

    def _load_symbols(self, vocab_path: Optional[str]) -> Any:
        # If your FS2 uses a symbol table, load it here. For now, fallback to basic charset.
        if vocab_path and os.path.exists(vocab_path):
            try:
                with open(vocab_path, "r", encoding="utf-8") as f:
                    return [ln.rstrip("\n") for ln in f if ln.strip()]
            except Exception:
                pass
        # basic ASCII letters + punctuation subset as placeholder
        return list(" _abcdefghijklmnopqrstuvwxyz'.,!?")

    def _text_to_sequence(self, text: str) -> list[int]:
        # Placeholder: map characters to indices within the symbol list
        # Replace with real G2P/phoneme tokenizer for your FS2.
        sym2id = {ch: i for i, ch in enumerate(self._symbols)}
        seq = []
        for ch in text.lower():
            seq.append(sym2id.get(ch, sym2id.get("_", 0)))
        # Add EOS token if your model expects it
        return seq

    # ---- Backend: ONNX ----
    def _infer_onnx(self, token_ids: list[int], pace: float) -> Any:
        assert self._ort_session is not None
        import numpy as np  # type: ignore

        # Input/Output names must match your exported FS2 graph.
        # Common patterns (adjust to your model):
        #   inputs:  "input_ids" (1, T), "pace" (1,), maybe "speaker_id"
        #   outputs: "mel" (1, n_mels, frames)
        input_ids = np.asarray([token_ids], dtype=np.int64)  # shape (1, T)
        pace_arr = np.asarray([pace], dtype=np.float32)      # shape (1,)

        # Try to be flexible about names
        name_map = {name: name for name in [i.name for i in self._ort_session.get_inputs()]}
        feed = {}
        if "input_ids" in name_map:
            feed[name_map["input_ids"]] = input_ids
        elif "text" in name_map:
            feed[name_map["text"]] = input_ids
        else:
            # Fallback: first input is token ids
            first = self._ort_session.get_inputs()[0].name
            feed[first] = input_ids

        # pace / speed control if present
        for cand in ("pace", "speed", "alpha"):
            if cand in name_map:
                feed[name_map[cand]] = pace_arr
                break

        try:
            outputs = self._ort_session.run(None, feed)
        except Exception as e:
            raise TTSSynthesisError(f"ONNX inference failed: {e}")

        # Heuristic: find mel by shape (B, n_mels, frames) or (B, frames, n_mels)
        mel = None
        for out in outputs:
            if hasattr(out, "ndim") and out.ndim == 3:
                _, a, c = out.shape  # batch_size, dim1, dim2
                if a == self._config.n_mels:
                    mel = out  # (1, n_mels, T)
                    break
                if c == self._config.n_mels:
                    mel = out.transpose(0, 2, 1)  # -> (1, n_mels, T)
                    break
        if mel is None:
            raise TTSSynthesisError("could not locate mel output in ONNX outputs")

        return mel  # keep as numpy array; vocoder adapter should accept it

    # ---- Backend: Torch ----
    def _build_torch_model_from_ckpt(self, ckpt: Any, device: str) -> Any:  # -> torch.nn.Module
        """Construct and load your FS2 model from the checkpoint.
        This is a stub hook—you must implement according to your FS2 repo.

        Args:
            ckpt: Loaded checkpoint data
            device: Target device for the model
        """
        # Suppress unused parameter warnings - this is an implementation stub
        _ = ckpt, device
        raise NotImplementedError("Implement FastSpeech2 model construction for your checkpoint format")

    def _infer_torch(self, token_ids: list[int], pace: float) -> Any:
        if self._torch_model is None:
            raise TTSSynthesisError("Torch model not loaded")
        try:
            import torch  # type: ignore
        except Exception as e:
            raise TTSSynthesisError(f"PyTorch not available: {e}")

        with torch.no_grad():
            ids = torch.tensor([token_ids], dtype=torch.long, device=self._config.device)
            pace_t = torch.tensor([pace], dtype=torch.float32, device=self._config.device)
            # Expected signature depends on your model. Common: model(ids, pace=pace_t)
            try:
                mel = self._torch_model(ids, pace=pace_t)  # shape (1, n_mels, T) or similar
            except TypeError:
                # Alternative arg name
                mel = self._torch_model(ids, alpha=pace_t)

        # If torch tensor, you may return tensor; downstream vocoder should accept torch or convert to numpy.
        return mel
