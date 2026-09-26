import logging
from collections import Counter
from pathlib import Path
import tempfile
import wave

from faster_whisper import WhisperModel

log = logging.getLogger("stt")
_models = {}


def get_model(name: str, device: str, compute_type: str):
    key = (name, device, compute_type)
    if key not in _models:
        _models[key] = WhisperModel(name, device=device, compute_type=compute_type)
    return _models[key]


def _looks_hallucinated(text: str, audio_seconds: float):
    clean = " ".join((text or "").split()).strip()
    if not clean:
        return True, "empty"

    words = clean.lower().split()

    # Whisper on noise/silence can emit extremely long repetitive text.
    # Natural telephone speech is usually far below this limit.
    max_chars = max(120, int(audio_seconds * 35 + 80))
    if len(clean) > max_chars:
        return True, f"too_long:{len(clean)}>{max_chars}"

    if len(words) >= 12:
        trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
        counts = Counter(trigrams)
        if counts and max(counts.values()) >= 4:
            return True, "repeated_trigram"

        unique_ratio = len(set(words)) / max(1, len(words))
        if len(words) >= 20 and unique_ratio < 0.32:
            return True, f"low_unique_ratio:{unique_ratio:.2f}"

    return False, ""


def transcribe_pcm16(
    pcm: bytes,
    model_name="small",
    device="cpu",
    compute_type="int8",
    sample_rate=8000,
):
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
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200,
            },
            beam_size=3,
            temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )

        text = " ".join(s.text.strip() for s in segments).strip()
        audio_seconds = len(pcm) / 2 / float(sample_rate)

        bad, reason = _looks_hallucinated(text, audio_seconds)
        if bad:
            if text:
                log.warning(
                    "Rejected STT hallucination (%s, %.2fs): %s",
                    reason,
                    audio_seconds,
                    text[:500],
                )
            return ""

        return text
    finally:
        path.unlink(missing_ok=True)
