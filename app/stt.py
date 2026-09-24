from functools import lru_cache
from pathlib import Path
import tempfile
import wave
from faster_whisper import WhisperModel

_models = {}

def get_model(name: str, device: str, compute_type: str):
    key = (name, device, compute_type)
    if key not in _models:
        _models[key] = WhisperModel(name, device=device, compute_type=compute_type)
    return _models[key]

def transcribe_pcm16(pcm: bytes, model_name="small", device="cpu", compute_type="int8", sample_rate=8000):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = Path(f.name)
    try:
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm)
        model = get_model(model_name, device, compute_type)
        segments, info = model.transcribe(
            str(path),
            language="pl",
            vad_filter=True,
            beam_size=3,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return text
    finally:
        path.unlink(missing_ok=True)
