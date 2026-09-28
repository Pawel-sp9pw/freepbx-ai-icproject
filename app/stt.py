import logging
import os
import re
from collections import Counter
from pathlib import Path
import tempfile
import wave

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt
from faster_whisper import WhisperModel

log = logging.getLogger("stt")
_models = {}

PROMPT_LEAK_PHRASES = (
    "oczekiwane odpowiedzi",
    "krótka odpowiedź na pytanie o potwierdzenie",
    "krotka odpowiedz na pytanie o potwierdzenie",
    "dzwoniący podaje nazwę swojej firmy po polsku",
    "dzwoniacy podaje nazwe swojej firmy po polsku",
    "dzwoniący opisuje problem techniczny lub usterkę po polsku",
    "dzwoniacy opisuje problem techniczny lub usterke po polsku",
)

KNOWN_WHISPER_HALLUCINATIONS = (
    "amara.org",
    "napisy stworzone",
    "napisy wykonane",
    "napisy przygotowane",
    "transkrypcja",
    "dziękuję za obejrzenie",
    "dziekuje za obejrzenie",
    "dziękuję za oglądanie",
    "dziekuje za ogladanie",
    "dzięki za oglądanie",
    "dzieki za ogladanie",
    "dzięki za obejrzenie",
    "dzieki za obejrzenie",
    "subskryb",
    "youtube.com",
    "youtu.be",
)


def get_model(name: str, device: str, compute_type: str, num_workers: int = 1):
    workers = max(1, min(4, int(num_workers or 1)))
    total_cpu = max(1, int(os.cpu_count() or 1))
    cpu_threads = min(8, max(1, total_cpu // workers)) if device == "cpu" else 0
    key = (name, device, compute_type, cpu_threads, workers)
    if key not in _models:
        kwargs = {
            "device": device,
            "compute_type": compute_type,
        }
        if device == "cpu":
            # CTranslate2 can execute several transcribe calls concurrently.
            # Split CPU threads between workers instead of letting one call
            # monopolize every vCPU.
            kwargs["cpu_threads"] = cpu_threads
            kwargs["num_workers"] = workers
        _models[key] = WhisperModel(name, **kwargs)
        log.info(
            "Loaded Whisper model %s on %s (%s), workers=%s, cpu_threads_per_worker=%s",
            name,
            device,
            compute_type,
            workers if device == "cpu" else "n/a",
            cpu_threads if device == "cpu" else "n/a",
        )
    return _models[key]


def _contact_language_reason(text: str):
    """Reject obvious English number-word transcripts in Polish phone mode."""
    words = set(re.findall(r"[a-z]+", (text or "").lower()))
    english_number_words = {
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
        "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
        "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
        "eighty", "ninety", "hundred",
    }
    if len(words & english_number_words) >= 2:
        return "non_polish_contact"
    return ""


def _confidence_floor_reason(score, mode: str):
    """Reject extremely weak non-confirmation candidates.

    Adaptive retry is allowed to work in the normal low-confidence range, but
    candidates below this floor are too unreliable to become ticket data.
    """
    if score is None or mode == "confirmation":
        return ""
    thresholds = {
        "company": -0.95,
        "problem": -0.95,
        "contact": -1.05,
        "normal": -1.05,
    }
    floor = thresholds.get(mode, -1.05)
    try:
        if float(score) < floor:
            return f"low_confidence:{float(score):.3f}<{floor:.2f}"
    except Exception:
        return ""
    return ""


def _looks_hallucinated(text: str, audio_seconds: float):
    clean = " ".join((text or "").split()).strip()
    if not clean:
        return True, "empty"

    lower_clean = clean.lower()
    words = lower_clean.split()

    if any(phrase in lower_clean for phrase in PROMPT_LEAK_PHRASES):
        return True, "prompt_leak"

    # Catch paraphrased leakage of the generic STT instruction, e.g.
    # "Dzwoniący podaje nazwę firmy, numer telefonu lub usterkę po polsku."
    # Whisper may slightly rewrite the prompt, so exact phrase matching is not enough.
    prompt_terms = (
        "nazwa firmy", "nazwę firmy", "numer telefonu", "telefon",
        "problem", "usterk", "po polsku",
    )
    if lower_clean.startswith(("dzwoniący podaje", "dzwoniacy podaje")):
        hits = sum(1 for term in prompt_terms if term in lower_clean)
        if hits >= 2:
            return True, "prompt_leak"

    if any(phrase in lower_clean for phrase in KNOWN_WHISPER_HALLUCINATIONS):
        return True, "known_whisper_hallucination"

    if "www." in lower_clean or "http://" in lower_clean or "https://" in lower_clean:
        return True, "url_hallucination"

    # Reject outputs that are effectively just one or more web addresses.
    urlish_tokens = re.findall(
        r"\b[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)+\b",
        lower_clean,
        flags=re.IGNORECASE,
    )
    if urlish_tokens:
        non_url = lower_clean
        for token in urlish_tokens:
            non_url = non_url.replace(token, " ")
        non_url = re.sub(r"[^a-ząćęłńóśźż0-9]+", " ", non_url, flags=re.IGNORECASE).strip()
        if not non_url:
            return True, "domain_only_hallucination"

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
            duration = max(
                0.1,
                float(getattr(s, "end", 0.0)) - float(getattr(s, "start", 0.0)),
            )
            weighted_sum += float(score) * duration
            total_weight += duration
        except Exception:
            continue
    return (weighted_sum / total_weight) if total_weight else None


def _prepare_phone_audio(pcm: bytes, sample_rate: int):
    """Prepare narrow-band telephone audio for Whisper.

    The application receives 8 kHz signed 16-bit PCM. We remove DC/very-low
    frequency rumble, gently normalize level and explicitly upsample to 16 kHz.
    Whisper would resample internally, but doing it here gives us predictable
    input and improves very quiet calls.
    """
    if not pcm:
        return pcm, sample_rate

    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    if samples.size < 32:
        return pcm, sample_rate

    # Remove DC offset first.
    samples -= float(np.mean(samples))

    # Telephone speech contains little useful information below ~90 Hz.
    # Keep the filter conservative so we do not damage consonants.
    try:
        nyquist = sample_rate / 2.0
        cutoff = min(90.0, nyquist * 0.1)
        if cutoff > 0 and samples.size > 128:
            sos = butter(2, cutoff / nyquist, btype="highpass", output="sos")
            samples = sosfiltfilt(sos, samples).astype(np.float32)
    except Exception:
        pass

    # Normalize quiet speech, but avoid amplifying near-silence/noise too much.
    rms = float(np.sqrt(np.mean(samples * samples) + 1e-12))
    peak = float(np.max(np.abs(samples)) + 1e-12)
    if rms >= 0.003:
        target_rms = 0.10
        gain = min(4.0, max(0.65, target_rms / rms))
        samples *= gain

    peak = float(np.max(np.abs(samples)) + 1e-12)
    if peak > 0.96:
        samples *= 0.96 / peak

    target_rate = 16000
    if sample_rate != target_rate:
        samples = resample_poly(samples, target_rate, sample_rate).astype(np.float32)
        sample_rate = target_rate

    samples = np.clip(samples, -0.98, 0.98)
    out = (samples * 32767.0).astype("<i2").tobytes()
    return out, sample_rate


def _needs_adaptive_retry(text: str, avg_logprob, mode: str):
    clean = " ".join((text or "").split()).strip()
    words = clean.split()

    if not clean:
        return True

    if mode == "company":
        # A second decode of a non-empty company name is very expensive on CPU
        # and, in practice, often returns the same name with no improvement.
        # Let the conversation layer handle low-confidence names with a quick
        # yes/no confirmation. Retry only when pass 1 produced no usable text
        # (empty / rejected hallucination).
        return not clean

    if mode == "contact":
        digits = re.sub(r"\D", "", clean)
        # A structurally valid phone number is already strong evidence.
        if 9 <= len(digits) <= 15:
            return False
        # Spoken phone numbers usually arrive as 5+ number words. A second
        # decode has repeatedly returned the same text, while the conversation
        # layer can normalize Polish number words deterministically.
        if len(words) >= 5:
            return False
        return True

    if mode == "problem":
        if avg_logprob is None:
            return False
        # Do not retry normal, clearly recognized descriptions. Short phrases
        # get a slightly stricter threshold because one wrong word matters more.
        threshold = -0.50 if len(words) <= 4 else -0.62
        return avg_logprob < threshold

    return False


def _choose_candidate(first_text, first_score, second_text, second_score):
    first_text = " ".join((first_text or "").split()).strip()
    second_text = " ".join((second_text or "").split()).strip()

    if not first_text:
        return second_text
    if not second_text:
        return first_text
    if first_text.lower() == second_text.lower():
        return first_text

    if second_score is not None and first_score is not None:
        if second_score >= first_score + 0.03:
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
    patience=1.0,
    max_new_tokens=None,
):
    kwargs = {
        "language": "pl",
        "vad_filter": bool(use_vad),
        "beam_size": int(beam_size),
        "patience": float(patience),
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "no_speech_threshold": 0.55,
        "log_prob_threshold": -1.2,
        "compression_ratio_threshold": 2.4,
        "initial_prompt": initial_prompt or None,
        "suppress_blank": True,
        "without_timestamps": True,
    }
    if max_new_tokens is not None:
        kwargs["max_new_tokens"] = int(max_new_tokens)
    if use_vad:
        kwargs["vad_parameters"] = {
            "min_silence_duration_ms": 350,
            "speech_pad_ms": 300,
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
    return_metadata=False,
    num_workers=1,
    log_context="",
):
    prepared_pcm, prepared_rate = _prepare_phone_audio(pcm, sample_rate)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = Path(f.name)

    try:
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(prepared_rate)
            w.writeframes(prepared_pcm)

        workers = max(1, min(4, int(num_workers or 1)))
        model = get_model(model_name, device, compute_type, workers)
        audio_seconds = len(pcm) / 2 / float(sample_rate)
        meta = {
            "mode": mode,
            "audio_seconds": round(audio_seconds, 3),
            "model": model_name,
            "workers": workers,
            "pass1": None,
            "pass2": None,
            "retry": False,
            "retry_reason": "",
            "selected": "",
            "selected_score": None,
        }

        first_text, first_score = _transcribe_once(
            model,
            path,
            initial_prompt,
            beam_size=5,
            # The application already segmented the utterance with WebRTC VAD.
            use_vad=False,
            patience=1.10,
            max_new_tokens=24 if mode == "company" else (48 if mode == "contact" else None),
        )
        first_bad, first_reason = _looks_hallucinated(first_text, audio_seconds)
        if not first_bad and mode == "contact":
            language_reason = _contact_language_reason(first_text)
            if language_reason:
                first_bad, first_reason = True, language_reason
        if not first_bad:
            confidence_reason = _confidence_floor_reason(first_score, mode)
            if confidence_reason:
                first_bad, first_reason = True, confidence_reason
        meta["pass1"] = {
            "text": first_text,
            "score": first_score,
            "rejected": bool(first_bad),
            "reason": first_reason or "",
        }
        if first_bad:
            if first_text:
                log.warning(
                    "%sRejected STT pass 1 (%s, %.2fs, score=%s): %s",
                    f"[{log_context}] " if log_context else "",
                    first_reason,
                    audio_seconds,
                    f"{first_score:.3f}" if first_score is not None else "n/a",
                    first_text[:500],
                )
            first_text = ""

        # Very short audio immediately after TTS often contains only residual
        # prompt leakage / subtitle hallucinations. A second heavy decode of the
        # same ~0.6 s fragment consistently produced another hallucination and
        # only consumed CPU. Drop it and wait for the caller's real utterance.
        if (
            first_bad
            and audio_seconds <= 0.80
            and first_reason in ("prompt_leak", "known_whisper_hallucination", "repeated_trigram")
        ):
            meta["retry"] = False
            meta["retry_reason"] = "short_rejected_audio"
            meta["selected"] = ""
            meta["selected_score"] = None
            return meta if return_metadata else ""

        # Confirmation remains deliberately conservative. Never use a second
        # decoding pass to manufacture a clearer "tak" from ambiguous audio.
        if mode == "confirmation":
            meta["selected"] = first_text
            meta["selected_score"] = first_score if first_text else None
            return meta if return_metadata else first_text

        if _needs_adaptive_retry(first_text, first_score, mode):
            meta["retry"] = True
            if not first_text:
                meta["retry_reason"] = first_reason or "empty_or_rejected"
            elif first_score is None:
                meta["retry_reason"] = "no_confidence"
            else:
                meta["retry_reason"] = "low_confidence_or_short_field"
            # For company/contact keep the domain hint. For a weak problem
            # description, remove the prompt in pass 2 to reduce prompt bias.
            retry_prompt = initial_prompt
            if mode in ("problem", "company"):
                retry_prompt = ""

            second_text, second_score = _transcribe_once(
                model,
                path,
                retry_prompt,
                beam_size=8,
                use_vad=False,
                patience=1.30,
                max_new_tokens=24 if mode == "company" else (48 if mode == "contact" else None),
            )
            second_bad, second_reason = _looks_hallucinated(second_text, audio_seconds)
            if not second_bad and mode == "contact":
                language_reason = _contact_language_reason(second_text)
                if language_reason:
                    second_bad, second_reason = True, language_reason
            if not second_bad:
                confidence_reason = _confidence_floor_reason(second_score, mode)
                if confidence_reason:
                    second_bad, second_reason = True, confidence_reason
            meta["pass2"] = {
                "text": second_text,
                "score": second_score,
                "rejected": bool(second_bad),
                "reason": second_reason or "",
            }
            if second_bad:
                if second_text:
                    log.warning(
                        "%sRejected STT pass 2 (%s, %.2fs, score=%s): %s",
                        f"[{log_context}] " if log_context else "",
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
                    "%sAdaptive STT %s: pass1=%r (%s), pass2=%r (%s), selected=%r",
                    f"[{log_context}] " if log_context else "",
                    mode,
                    first_text,
                    f"{first_score:.3f}" if first_score is not None else "n/a",
                    second_text,
                    f"{second_score:.3f}" if second_score is not None else "n/a",
                    selected,
                )
            selected_score = None
            if selected:
                # When both passes returned identical text, _choose_candidate
                # intentionally keeps pass 1; preserve its confidence too.
                if first_text and second_text and first_text.lower() == second_text.lower():
                    selected_score = first_score
                elif second_text and selected == second_text:
                    selected_score = second_score
                elif first_text and selected == first_text:
                    selected_score = first_score
            meta["selected"] = selected
            meta["selected_score"] = selected_score
            return meta if return_metadata else selected

        meta["selected"] = first_text
        meta["selected_score"] = first_score if first_text else None
        return meta if return_metadata else first_text
    finally:
        path.unlink(missing_ok=True)
