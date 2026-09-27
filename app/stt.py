import logging
from collections import Counter
from pathlib import Path
import tempfile
import wave

from faster_whisper import WhisperModel

log = logging.getLogger("stt")
_models = {}

PROMPT_LEAK_PHRASES = (
    "oczekiwane odpowiedzi",
    "krótka odpowiedź na pytanie o potwierdzenie",
    "krotka odpowiedz na pytanie o potwierdzenie",
)


def get_model(name: str, device: str, compute_type: str):
    key = (name, device, compute_type)
    if key not in _models:
        _models[key] = WhisperModel(name, device=device, compute_type=compute_type)
    return _models[key]


def _looks_hallucinated(text: str, audio_seconds: float):
    clean = " ".join((text or "").split()).strip()
    if not clean:
        return True, "empty"

    lower_clean = clean.lower()
    words = lower_clean.split()

    if any(phrase in lower_clean for phrase in PROMPT_LEAK_PHRASES):
        return True, "prompt_leak"

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


def _segment_confidence(segments):
    weighted_sum = 0.0
    total_weight = 0.0
    for s in segments:
        score = getattr(s, "avg_logprob", None)
        if score is None:
            continue
        try:
            duration = max(0.1, float(getattr(s, "end", 0.0)) - float(getattr(s, "start", 0.0)))
            weighted_sum += float(score) * duration
            total_weight += duration
        except Exception:
            continue
    return (weighted_sum / total_weight) if total_weight else None


def _needs_adaptive_retry(text: str, avg_logprob, mode: str):
    if mode != "company":
        return False

    clean = " ".join((text or "").split()).strip()
    if not clean:
        return True

    words = clean.split()
    if len(words) > 3:
        return False

    # Short company names are the hardest case on 8 kHz telephony. Run a
    # precision pass for them even when Whisper is moderately confident.
    if avg_logprob is None:
        return True
    return avg_logprob < -0.20 or len(words) <= 2


def _choose_candidate(first_text, first_score, second_text, second_score):
    first_text = " ".join((first_text or "").split()).strip()
    second_text = " ".join((second_text or "").split()).strip()

    if not first_text:
        return second_text
    if not second_text:
        return first_text
    if first_text.lower() == second_text.lower():
        return first_text

    # Prefer the precision pass only when it is measurably more confident.
    if second_score is not None and first_score is not None:
        if second_score >= first_score + 0.04:
            return second_text
        return first_text

    if second_score is not None and first_score is None:
        return second_text
    return first_text


def _transcribe_once(
    model,
    path,
    initial_prompt,
    beam_size,
    use_vad=True,
):
    kwargs = {
        "language": "pl",
        "vad_filter": bool(use_vad),
        "beam_size": int(beam_size),
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "no_speech_threshold": 0.6,
        "log_prob_threshold": -1.0,
        "compression_ratio_threshold": 2.4,
        "initial_prompt": initial_prompt or None,
    }
    if use_vad:
        kwargs["vad_parameters"] = {
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,
        }

    segments, _info = model.transcribe(str(path), **kwargs)
    segments = list(segments)
    text = " ".join(s.text.strip() for s in segments).strip()
    return text, _segment_confidence(segments)


def transcribe_pcm16(
    pcm: bytes,
    model_name="small",
    device="cpu",
    compute_type="int8",
    sample_rate=8000,
    initial_prompt="",
    mode="normal",
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
        audio_seconds = len(pcm) / 2 / float(sample_rate)

        first_text, first_score = _transcribe_once(
            model,
            path,
            initial_prompt,
            beam_size=3,
            use_vad=True,
        )
        first_bad, first_reason = _looks_hallucinated(first_text, audio_seconds)
        if first_bad:
            if first_text:
                log.warning(
                    "Rejected STT pass 1 (%s, %.2fs, score=%s): %s",
                    first_reason,
                    audio_seconds,
                    f"{first_score:.3f}" if first_score is not None else "n/a",
                    first_text[:500],
                )
            first_text = ""

        # Confirmation is deliberately single-pass: we never want a precision
        # retry to turn an unclear utterance into an artificial "tak".
        if mode == "confirmation":
            return first_text

        if _needs_adaptive_retry(first_text, first_score, mode):
            second_text, second_score = _transcribe_once(
                model,
                path,
                initial_prompt,
                beam_size=5,
                # Audio has already been segmented by the application VAD.
                # Disabling Whisper VAD here helps avoid clipping short names.
                use_vad=False,
            )
            second_bad, second_reason = _looks_hallucinated(second_text, audio_seconds)
            if second_bad:
                if second_text:
                    log.warning(
                        "Rejected STT pass 2 (%s, %.2fs, score=%s): %s",
                        second_reason,
                        audio_seconds,
                        f"{second_score:.3f}" if second_score is not None else "n/a",
                        second_text[:500],
                    )
                second_text = ""

            selected = _choose_candidate(
                first_text,
                first_score,
                second_text,
                second_score,
            )
            if second_text and second_text != first_text:
                log.info(
                    "Adaptive STT company: pass1=%r (%s), pass2=%r (%s), selected=%r",
                    first_text,
                    f"{first_score:.3f}" if first_score is not None else "n/a",
                    second_text,
                    f"{second_score:.3f}" if second_score is not None else "n/a",
                    selected,
                )
            return selected

        return first_text
    finally:
        path.unlink(missing_ok=True)
