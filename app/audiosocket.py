import asyncio
import json
import logging
import re
import struct
import time
import unicodedata
import uuid
from collections import Counter, deque
from difflib import SequenceMatcher
import webrtcvad

from .config import load_settings, decrypt_secret
from .stt import transcribe_pcm16
from .tts import synthesize_pcm8k
from .llm import ask_ollama, interpret_turn, looks_like_prompt_injection
from .icproject import ICProjectClient
from .monitoring import call_started, add_message, finish_call
from .call_registry import consume_caller

log = logging.getLogger("audiosocket")

TYPE_HANGUP = 0x00
TYPE_UUID = 0x01
TYPE_DTMF = 0x03
TYPE_PCM_8K = 0x10

async def read_exactly_or_none(reader, n):
    try:
        return await reader.readexactly(n)
    except (asyncio.IncompleteReadError, ConnectionResetError):
        return None

async def read_packet(reader):
    hdr = await read_exactly_or_none(reader, 3)
    if not hdr:
        return None, None
    typ = hdr[0]
    length = int.from_bytes(hdr[1:3], "big")
    payload = await read_exactly_or_none(reader, length)
    if payload is None:
        return None, None
    return typ, payload

async def send_packet(writer, typ, payload=b""):
    if writer.is_closing():
        return False
    try:
        writer.write(bytes([typ]) + len(payload).to_bytes(2, "big") + payload)
        await writer.drain()
        return True
    except (ConnectionResetError, BrokenPipeError, RuntimeError):
        return False

async def send_pcm(writer, pcm: bytes):
    # 20 ms @ 8 kHz, mono, 16-bit = 320 bytes
    for i in range(0, len(pcm), 320):
        chunk = pcm[i:i+320]
        if len(chunk) < 320:
            chunk += b"\x00" * (320 - len(chunk))
        ok = await send_packet(writer, TYPE_PCM_8K, chunk)
        if not ok:
            break
        await asyncio.sleep(0.02)


def parse_customer_directory(raw: str):
    items = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            name, phone = line.split("|", 1)
        else:
            name, phone = line, ""
        name = name.strip()
        phone_digits = re.sub(r"\D", "", phone or "")
        if name:
            items.append({"name": name, "phone": phone_digits})
    return items


def normalize_company(value: str):
    text = (value or "").lower()
    # Normalize Polish diacritics and common conversational prefixes so that
    # short STT variants can be compared fairly with the customer directory.
    text = text.translate(str.maketrans({
        "ą": "a", "ć": "c", "ę": "e", "ł": "l",
        "ń": "n", "ó": "o", "ś": "s", "ź": "z", "ż": "z",
    }))
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(
        r"\b(?:dzien dobry|dobry wieczor|czesc|witam|firma|spolka|tu|z tej strony)\b",
        " ",
        text,
    )
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def clean_company_display_name(value: str):
    """Remove conversational prefixes without changing the actual company name."""
    text = company_without_phone(value or "")
    text = re.sub(r"^\s*(?:dzień dobry|dzien dobry|dobry wieczór|dobry wieczor|cześć|czesc|witam)[,.:;\-\s]*", "", text, flags=re.IGNORECASE)
    # Prefixes may be stacked: "Dzień dobry, tu firma Alfatest".
    for _ in range(3):
        cleaned = re.sub(r"^\s*(?:tu|firma|spółka|spolka|z tej strony)\b[,.:;\-\s]*", "", text, flags=re.IGNORECASE)
        if cleaned == text:
            break
        text = cleaned
    return re.sub(r"\s+", " ", text).strip(" ,.;:-")


def speak_phone(value: str):
    digits = re.sub(r"\D", "", value or "")
    return " ".join(digits) if digits else value


def _spoken_polish_number_digits(value: str):
    """Convert Polish spoken/mixed phone-number text to digits conservatively."""
    text = (value or "").lower()
    text = text.translate(str.maketrans({
        "ą": "a", "ć": "c", "ę": "e", "ł": "l",
        "ń": "n", "ó": "o", "ś": "s", "ź": "z", "ż": "z",
    }))
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    # Harmless conversational words commonly emitted around a phone number.
    fillers = {
        "moj", "moja", "numer", "telefon", "telefonu", "kontaktowy", "kontaktowego",
        "to", "jest", "prosze", "podaje", "podam", "tak", "brzmi",
    }
    aliases = {
        "szescsty": "szescset",
        "szescty": "szescset",
        # Common Whisper substitutions of ordinal forms while callers dictate
        # phone-number groups.
        "jedenasty": "jedenascie",
        "jedenaste": "jedenascie",
        "dwunasty": "dwanascie",
        "trzynasty": "trzynascie",
        "czternasty": "czternascie",
        "pietnasty": "pietnascie",
        "szesnasty": "szesnascie",
        "siedemnasty": "siedemnascie",
        "osiemnasty": "osiemnascie",
        "osianasty": "osiemnascie",
        "dziewietnasty": "dziewietnascie",
        "dwudziesta": "dwadziescia",
    }
    units = {
        "zero": 0, "jeden": 1, "jedna": 1, "jedno": 1,
        "dwa": 2, "dwie": 2, "trzy": 3, "trzech": 3, "czy": 3, "cztery": 4,
        "piec": 5, "szesc": 6, "siedem": 7, "osiem": 8, "dziewiec": 9,
    }
    teens = {
        "dziesiec": 10, "jedenascie": 11, "dwanascie": 12, "trzynascie": 13,
        "czternascie": 14, "pietnascie": 15, "szesnascie": 16,
        "siedemnascie": 17, "osiemnascie": 18, "dziewietnascie": 19,
    }
    tens = {
        "dwadziescia": 20, "dwadziesta": 20, "dwudziestu": 20,
        "trzydziesci": 30, "trzydziestu": 30,
        "czterdziesci": 40, "czterdziestu": 40,
        "piecdziesiat": 50, "piecdziesieciu": 50,
        "szescdziesiat": 60, "szescdziesieciu": 60,
        "siedemdziesiat": 70, "siedemdziesieciu": 70,
        "osiemdziesiat": 80, "osiemdziesieciu": 80,
        "dziewiecdziesiat": 90, "dziewiecdziesieciu": 90,
    }
    hundreds = {
        "sto": 100, "stu": 100, "dwiescie": 200, "trzysta": 300, "czterysta": 400,
        "piecset": 500, "szescset": 600, "siedemset": 700,
        "osiemset": 800, "dziewiecset": 900,
    }

    tokens = re.findall(r"\d+|[a-z]+", text)
    if not tokens:
        return ""
    tokens = [aliases.get(token, token) for token in tokens if token not in fillers]
    known_words = set(units) | set(teens) | set(tens) | set(hundreds)
    if any(not token.isdigit() and token not in known_words for token in tokens):
        return ""

    groups = []
    current = 0
    has_hundred = False
    has_tens = False
    has_unit = False

    def flush():
        nonlocal current, has_hundred, has_tens, has_unit
        if has_hundred or has_tens or has_unit:
            groups.append(str(current))
        current = 0
        has_hundred = has_tens = has_unit = False

    for token in tokens:
        if token.isdigit():
            flush()
            groups.append(token)
        elif token in hundreds:
            if has_hundred or has_tens or has_unit:
                flush()
            current = hundreds[token]
            has_hundred = True
        elif token in teens:
            if has_tens or has_unit:
                flush()
            current += teens[token]
            has_tens = True
            has_unit = True
        elif token in tens:
            if has_tens or has_unit:
                flush()
            current += tens[token]
            has_tens = True
        else:
            if token == "zero":
                flush()
                groups.append("0")
                continue
            if has_unit:
                flush()
            current += units[token]
            has_unit = True

    flush()
    return "".join(groups)


def extract_phone_digits(value: str, mode: str = "pl"):
    """Normalize a typed or spoken contact number according to validation."""
    mode = str(mode or "pl").lower()
    digits = _spoken_polish_number_digits(value)
    if not digits:
        digits = re.sub(r"\D", "", value or "")

    if mode == "international":
        return digits if 7 <= len(digits) <= 15 else ""

    if len(digits) == 9:
        return digits
    if len(digits) == 11 and digits.startswith("48"):
        return digits[2:]
    if len(digits) == 13 and digits.startswith("0048"):
        return digits[4:]
    return ""


def looks_like_invalid_company_name(value: str):
    normalized = " ".join((value or "").lower().strip(" .,!?:;").split())
    if not normalized:
        return True

    bad_fragments = (
        "dzwoniący podaje nazwę swojej firmy",
        "dzwoniacy podaje nazwe swojej firmy",
        "amara.org",
        "youtube.com",
        "youtu.be",
        "napisy stworzone",
        "napisy wykonane",
        "napisy przygotowane",
        "transkrypcja",
        "dziękuję za obejrzenie",
        "dziekuje za obejrzenie",
        "dziękuję za oglądanie",
        "dziekuje za ogladanie",
        "subskryb",
    )
    if any(fragment in normalized for fragment in bad_fragments):
        return True

    # Short acknowledgements, farewells and filler-only transcripts cannot be
    # valid company names. Keep this check company-specific so confirmation
    # utterances such as "nie" remain valid in their own state.
    company_only_noise = {
        "tak", "nie", "dobrze", "dziękuję", "dziekuje", "bardzo dziękuję",
        "bardzo dziekuje", "dzień dobry", "dzien dobry", "cześć", "czesc",
        "do zobaczenia", "do widzenia", "co to jest", "dzwonię", "dzwonie",
        "no", "to jest", "yyy", "yyy yyy", "hmm",
    }
    if normalized in company_only_noise:
        return True

    words = re.findall(r"[a-ząćęłńóśźż0-9]+", normalized, flags=re.IGNORECASE)
    if words:
        counts = Counter(words)
        most_common = counts.most_common(1)[0][1]
        # Typical Whisper loop: "nie, nie, nie..." or "no, no, no...".
        if len(words) >= 4 and len(counts) <= 2 and most_common >= 4:
            return True
        # Filler-heavy fragments such as "no, no, no, to jest".
        filler_words = {"no", "to", "jest", "yyy", "hmm"}
        if len(words) >= 3 and all(word in filler_words for word in words):
            return True

    if "www." in normalized or "http://" in normalized or "https://" in normalized:
        return True

    # A transcript consisting only of a domain/address is not a company name.
    if re.fullmatch(r"[a-z0-9.-]+\.(?:pl|com|eu|org|net)(?:\s+[a-z0-9.-]+\.(?:pl|com|eu|org|net))*", normalized):
        return True

    return False


def looks_like_ticket_cancellation(text: str):
    """Detect an explicit request not to create / to abandon the ticket."""
    normalized = " ".join((text or "").lower().strip(" .,!?:;").split())
    if not normalized:
        return False

    patterns = (
        r"\bnie\s+(?:prosz[ęe]\s+)?(?:zak[łl]ada[ćc]|zak[łl]adaj|tw[oó]rz|tw[oó]rzcie|rejestruj|zapisuj)\b.*\bzg[łl][ou]szen",
        r"\bprosz[ęe]\s+(?:o\s+)?nie\s+(?:zak[łl]ada[ćc]|zak[łl]adaj|tw[oó]rz|rejestruj|zapisuj)\b.*\bzg[łl][ou]szen",
        r"\b(anuluj|anulowa[ćc]|wycofuj[ęe]|wycofaj|rezygnuj[ęe]|prozygnuj[ęe])\b.*\bzg[łl][ou]szen",
        r"\bzg[łl][ou]szeni[ea]\b.*\b(?:ju[żz]\s+)?nie\s+(?:jest\s+)?potrzebn",
        r"\bzg[łl][ou]szenie\b.*\b(niepotrzebne|nieaktualne|anuluj|wycofaj)\b",
        r"\bju[żz]\s+(?:zacz[ęe][łl]o\s+)?dzia[łl]a[ćc]?\b.*\b(?:nie|unie)\s+.*\bzg[łl][ou]szen",
        r"\bprosz[ęe]\b.*\b(?:nie|unie)\s+(?:zak[łl]ada[ćc]|zak[łl]adaj|tw[oó]rz|rejestruj|zapisuj)\b.*\bzg[łl][ou]szen",
        r"\b(?:anul|omu[łl])\w*\b.*\bzg[łl][ou]szen",
        r"\bprosz[ęe]\s+anulowa[ćc]\b.*\b(?:zg[łl][ou]szen|z[łl][ou]szen)",
        r"\bnie\s+(?:za[łl]atwia[ćc]|za[łl]atwiaj)\b.*\b(?:zg[łl][ou]szen|z[łl][ou]szen)",
        r"\b(?:zg[łl][ou]szen|z[łl][ou]szen)\w*\b.*\bnie\s+(?:jest\s+)?potrzebn",
        r"\bproblem\s+(?:ju[żz]\s+)?(?:rozwi[aą]zany|znikn[aą][łl]|ust[aą]pi[łl])\b.*\bnie\s+.*\bzg[łl][ou]szen",
    )
    return any(re.search(pattern, normalized, re.IGNORECASE) for pattern in patterns)


def looks_like_abusive_dismissal(text: str):
    """Detect a hostile/dismissive utterance that means the caller is done.

    Do not treat profanity itself as a problem description. If the same
    utterance contains a concrete technical symptom, keep it as a valid
    service problem.
    """
    normalized = " ".join((text or "").lower().strip(" .,!?:;").split())
    if not normalized:
        return False

    problem_signals = (
        "nie działa", "nie dziala", "nie mogę", "nie moge", "nie można", "nie mozna",
        "błąd", "blad", "awaria", "usterka", "problem z", "brak ", "wyskakuje",
        "zawiesza", "rozłącza", "rozlacza", "nie otwiera", "nie drukuje",
        "nie loguje", "nie zapisuje", "nie wysyła", "nie wysyla", "przestał",
        "przestal", "zepsuł", "zepsul",
    )
    if any(signal in normalized for signal in problem_signals):
        return False

    patterns = (
        # Common profanity variants and typical Polish STT confusions.
        r"\bspier(?:dal|dol|dziel)aj\b",
        r"\bspier(?:dal|dol|dziel)\s+si[ęe]\b",
        r"\bspieprz(?:aj)?\b",
        r"\bspadaj\b",
        r"\bspad[aą]j\b",
        r"\bodczep\s+si[ęe]\b",
        r"\bodczep\s+ode\s+mnie\b",
        r"\bodwal\s+si[ęe]\b",
        r"\bodwal\s+ode\s+mnie\b",
        r"\bodp(?:ieprz|ierdol|iepsz)\s+si[ęe]\b",
        r"\bodp(?:ieprz|ierdol|iepsz)\s+ode\s+mnie\b",
        r"\bpieprz\s+si[ęe]\b",
        r"\bwal\s+si[ęe]\b",
        r"\bwon\b",
        r"\bdaj\s+(mi\s+)?spok[oó]j\b",
        r"\bzostaw\s+mnie\b",
        r"\bnie\s+chc[ęe]\s+(ju[żz]\s+)?(z\s+tob[aą]\s+)?rozmawia[ćc]\b",
        r"\bnie\s+b[ęe]d[ęe]\s+(z\s+tob[aą]\s+)?rozmawia[ćc]\b",
        r"\bnie\s+dzwo[ńn]\b",
        r"\bprosz[ęe]\s+nie\s+dzwo[ńn]\b",
        r"\bko[ńn]cz\b",
        r"\bsko[ńn]cz\s+(ju[żz]\s+)?(t[ęe]\s+)?rozmow[ęe]\b",
        r"\broz[łl][ąa]cz\s+si[ęe]\b",
        r"\broz[łl][ąa]cz\b",
    )
    return any(re.search(pattern, normalized, re.IGNORECASE) for pattern in patterns)


def looks_like_human_handoff_request(text: str):
    """Detect requests to speak with a human instead of a service problem."""
    normalized = " ".join((text or "").lower().strip(" .,!?:;").split())
    if not normalized:
        return False

    # If the same utterance also contains a concrete fault/symptom, keep the
    # technical content as a valid problem description.
    problem_signals = (
        "nie działa", "nie dziala", "nie mogę", "nie moge", "nie można", "nie mozna",
        "błąd", "blad", "awaria", "usterka", "problem z", "brak ", "wyskakuje",
        "zawiesza", "rozłącza", "rozlacza", "nie otwiera", "nie drukuje",
        "nie loguje", "nie zapisuje", "nie wysyła", "nie wysyla", "przestał",
        "przestal", "zepsuł", "zepsul",
    )
    if any(signal in normalized for signal in problem_signals):
        return False

    patterns = (
        r"\b(chc[ęe]|chcia[łl]bym|prosz[ęe])\b.*\b(porozmawia[ćc]|rozmawia[ćc])\b.*\b(cz[łl]owiek|konsultant|operator|serwisant|pracownik)",
        r"\b(po[łl][ąa]cz|prze[łl][ąa]cz|przekieruj)\b.*\b(cz[łl]owiek|konsultant|operator|serwis|serwisant|pracownik)",
        r"\b(chc[ęe]|poprosz[ęe])\b.*\b(cz[łl]owieka|konsultanta|operatora|serwisanta|pracownika)",
        r"\b(cz[łl]owiek|konsultant|operator|serwisant)\b",
    )
    return any(re.search(pattern, normalized, re.IGNORECASE) for pattern in patterns)


def looks_like_ticket_meta_request(text: str):
    """Return True when the caller talks about creating/routing the ticket
    instead of describing the actual service problem.
    """
    normalized = " ".join((text or "").lower().strip(" .,!?:;").split())
    if not normalized:
        return False

    # If there is an actual symptom/error in the same utterance, keep it as a
    # valid problem description even if the caller also says "utwórz zgłoszenie".
    problem_signals = (
        "nie działa", "nie dziala", "nie mogę", "nie moge", "nie można", "nie mozna",
        "błąd", "blad", "awaria", "usterka", "problem z", "brak ", "wyskakuje",
        "zawiesza", "rozłącza", "rozlacza", "wolno działa", "wolno dziala",
        "nie otwiera", "nie drukuje", "nie loguje", "nie zapisuje", "nie wysyła",
        "nie wysyla", "przestał", "przestal", "zepsuł", "zepsul",
    )
    if any(signal in normalized for signal in problem_signals):
        return False

    meta_patterns = (
        r"\b(utw[oó]rz|stw[oó]rz|zapisz|dodaj|za[łl][oó][żz]|przyjmij|zarejestruj)\b.*\bzg[łl][ou]szen",
        r"\b(przeka[żz]|wy[śs]lij|prze[śs]lij)\b.*\b(serwis|zg[łl][ou]szen)",
        r"\bzg[łl]o[śs]\b.*\b(serwis|to|spraw[ęe])",
        r"\b(testowe|testowy|test)\b.*\bzg[łl][ou]szen",
        r"\bzg[łl][ou]szenie\b.*\b(serwis|utw[oó]rz|zapisz|przeka[żz])",
    )
    return any(re.search(pattern, normalized, re.IGNORECASE) for pattern in meta_patterns)


def matches_confirmation_phrase(text: str, phrases: tuple[str, ...]):
    normalized = " ".join((text or "").lower().strip(" .,!?:;").split())
    if not normalized:
        return False
    if normalized in phrases:
        return True

    # Whisper often repeats a short confirmation ("nie, nie, nie") or appends
    # a harmless acknowledgement ("tak, dobra"). Collapse only unambiguous
    # repeated yes/no forms; never infer intent from mixed yes+no text.
    words = re.findall(r"[a-ząćęłńóśźż]+", normalized, flags=re.IGNORECASE)
    if not words:
        return False
    has_yes = "tak" in words
    has_no = "nie" in words
    if has_yes and has_no:
        return False

    phrase_set = set(phrases)
    if has_yes and "tak" in phrase_set:
        allowed_yes_tail = {"tak", "dobra", "dobrze", "zgadza", "się", "sie"}
        return all(word in allowed_yes_tail for word in words)
    if has_no and "nie" in phrase_set:
        return all(word == "nie" for word in words)
    return False


def apply_llm_fill_only(ticket_data: dict, ticket_update: dict):
    """Apply only missing user-facing ticket fields from LLM output.

    Existing values are immutable here. Explicit correction flow is the only
    place allowed to replace them.
    """
    allowed_fill_keys = {"company", "contact", "title", "description"}
    for key, value in (ticket_update or {}).items():
        if key not in allowed_fill_keys:
            continue
        if value in (None, "", [], {}):
            continue
        if key == "description" and (
            looks_like_ticket_meta_request(str(value))
            or looks_like_human_handoff_request(str(value))
            or looks_like_abusive_dismissal(str(value))
            or looks_like_ticket_cancellation(str(value))
        ):
            continue
        if key == "company" and looks_like_invalid_company_name(str(value)):
            continue
        if ticket_data.get(key):
            continue
        ticket_data[key] = value
    return ticket_data


def company_without_phone(value: str):
    cleaned = re.sub(r"[\d\s,.;:+()\-]{7,}", " ", value or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:-")
    return cleaned.strip()


def extract_problem_fragment(text: str):
    """Extract an explicit technical-problem fragment from a mixed utterance.

    Used when the caller gives company/name and problem in the same sentence,
    before the state machine formally asks for the problem. This is deliberately
    conservative: it only triggers on concrete fault/symptom phrases.
    """
    source = " ".join((text or "").strip().split())
    if not source:
        return ""

    signals = (
        "nie działa", "nie dziala", "nie mogę", "nie moge", "nie można", "nie mozna",
        "błąd", "blad", "awaria", "usterka", "problem z", "brak ", "wyskakuje",
        "zawiesza", "rozłącza", "rozlacza", "wolno działa", "wolno dziala",
        "nie otwiera", "nie drukuje", "nie loguje", "nie zapisuje", "nie wysyła",
        "nie wysyla", "przestał", "przestal", "zepsuł", "zepsul",
    )
    lower = source.lower()
    positions = [lower.find(signal) for signal in signals if lower.find(signal) >= 0]
    if not positions:
        return ""

    start = min(positions)

    # Prefer the whole sentence containing the first explicit symptom.
    sentence_start = max(
        source.rfind(".", 0, start),
        source.rfind("!", 0, start),
        source.rfind("?", 0, start),
        source.rfind(";", 0, start),
    )
    if sentence_start >= 0:
        start = sentence_start + 1

    fragment = source[start:].strip(" ,.;:-")
    if len(fragment) < 3:
        return ""
    return fragment


def match_customer(company: str, contact: str, directory: list):
    contact_digits = re.sub(r"\D", "", contact or "")
    # Phone/CallerID is authoritative and always wins over fuzzy name matching.
    if contact_digits:
        for item in directory:
            phone = re.sub(r"\D", "", item.get("phone", "") or "")
            if not phone:
                continue
            exact = contact_digits == phone
            suffix = (
                len(contact_digits) >= 9
                and len(phone) >= 9
                and (
                    contact_digits.endswith(phone[-9:])
                    or phone.endswith(contact_digits[-9:])
                )
            )
            if exact or suffix:
                return item, 1.0

    source = normalize_company(company)
    if not source:
        return None, 0.0

    candidates = []
    for item in directory:
        target = normalize_company(item.get("name", ""))
        if not target:
            continue

        score = SequenceMatcher(None, source, target).ratio()
        compact_source = source.replace(" ", "")
        compact_target = target.replace(" ", "")
        if compact_source and compact_target:
            score = max(
                score,
                SequenceMatcher(None, compact_source, compact_target).ratio(),
            )

        # Exact token containment is a strong signal for inputs such as
        # "Firma Paweł" vs "Paweł", after conversational prefixes are removed.
        if source == target:
            score = 1.0
        elif source in target or target in source:
            score = max(score, 0.90)

        candidates.append((score, item))

    if not candidates:
        return None, 0.0

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    best_score, best = candidates[0]
    second_score = candidates[1][0] if len(candidates) > 1 else 0.0
    margin = best_score - second_score

    # Short names are especially prone to Whisper substitutions (Atul/Ator
    # for Artur). Allow a slightly lower score only when the best candidate
    # clearly beats every alternative. Otherwise ask for confirmation.
    compact_len = len(source.replace(" ", ""))
    if compact_len <= 6:
        accepted = best_score >= 0.62 and margin >= 0.18
    else:
        accepted = best_score >= 0.72 and margin >= 0.12

    return (best, best_score) if accepted else (None, best_score)


class CallSession:
    def __init__(self, call_id: str, writer):
        self.call_id = call_id
        self.writer = writer
        self.settings = load_settings()
        self.history = []
        self.vad = webrtcvad.Vad(2)
        self.frame_buf = bytearray()
        self.speech = bytearray()
        self.pre_roll = deque(maxlen=10)
        self.speaking = False
        self.silence_frames = 0
        self.turns = 0
        self.closed = False
        self.ticket_data = {}
        self.ticket_field_meta = {}
        self.customer_directory = parse_customer_directory(self.settings.get("customer_directory", ""))
        self.confirmation_pending = False
        self.awaiting_correction = False
        self.correction_field = ""
        self.correction_attempts = 0
        self.original_caller = ""
        self.caller_matched_customer = False
        self.awaiting_company = False
        self.awaiting_contact = False
        self.awaiting_problem = False
        self.company_confirmation_pending = False
        self.company_candidate = ""
        self.company_candidate_score = None
        self.company_confirmation_context = ""
        self.company_candidate_phone = ""
        self.rejected_company_names = set()
        self.early_problem_score = None
        self.company_trusted = False
        self.contact_trusted = False
        self.contact_attempts = 0
        self.awaiting_contact_dtmf = False
        self.dtmf_contact_buffer = ""
        self.dtmf_contact_context = ""
        self.stt_misses = 0
        self.listen_not_before = 0.0
        self.last_tts_end = 0.0
        self.last_repeat_prompt = 0.0
        self.confirmation_misses = 0
        self.last_question = ""
        self.final_status = "ended"
        self.ticket_ref = ""
        self.final_error = ""

    def mark_ticket_field(self, field, source, score=None, trusted=None):
        meta = {"source": str(source or "unknown")}
        if score is not None:
            try:
                meta["score"] = round(float(score), 4)
            except (TypeError, ValueError):
                pass
        if trusted is not None:
            meta["trusted"] = bool(trusted)
        self.ticket_field_meta[str(field)] = meta

    def record_customer_match(self, source, company, contact, matched_customer, score):
        """Persist safe matching telemetry for post-call regression analysis."""
        payload = {
            "source": str(source or ""),
            "input_company": str(company or ""),
            "normalized_company": normalize_company(str(company or "")),
            "input_contact": re.sub(r"\D", "", str(contact or "")),
            "matched": bool(matched_customer),
            "matched_name": str((matched_customer or {}).get("name", "") or ""),
            "score": round(float(score or 0.0), 4),
            "company_trusted": bool(self.company_trusted),
            "contact_trusted": bool(self.contact_trusted),
        }
        try:
            add_message(
                self.call_id,
                "match_debug",
                json.dumps(payload, ensure_ascii=False),
            )
        except Exception:
            # Diagnostic telemetry must never break the active phone call.
            log.warning("[%s] Could not persist customer match telemetry", self.call_id, exc_info=True)

    async def say(self, text):
        log.info("[%s] TTS: %s", self.call_id, text)
        add_message(self.call_id, "assistant", text)
        pcm = await synthesize_pcm8k(
            self.settings["piper_url"],
            text,
            self.settings.get("piper_voice"),
        )
        await send_pcm(self.writer, pcm)
        self.frame_buf.clear()
        self.speech.clear()
        self.pre_roll.clear()
        self.speaking = False
        self.silence_frames = 0
        self.stt_misses = 0
        self.last_tts_end = time.monotonic()
        self.listen_not_before = self.last_tts_end + (0.20 if (self.confirmation_pending or self.company_confirmation_pending) else 0.45)

    async def say_confirmation_summary(self):
        company = str(self.ticket_data.get("company", "") or "").strip() or "nie podano"
        contact = str(self.ticket_data.get("contact", "") or "").strip() or "nie podano"
        spoken_contact = speak_phone(contact) if contact != "nie podano" else contact
        description = str(self.ticket_data.get("description", "") or "").strip() or "nie podano"
        if description != "nie podano":
            description = description.rstrip(" .!?")

        await self.say(
            "Podsumuję zgłoszenie. "
            f"Firma: {company}. "
            f"Numer kontaktowy: {spoken_contact}. "
            f"Problem: {description}."
        )
        await asyncio.sleep(0.35)
        await self.say("Proszę powiedzieć tak, jeśli dane są poprawne, albo nie, jeśli wymagają poprawy.")

    def has_complete_ticket_data(self):
        return all(
            str(self.ticket_data.get(key, "") or "").strip()
            for key in ("company", "contact", "description")
        )

    def problem_confidence_is_high(self, selected_score):
        """Return True only for a strongly recognized problem description."""
        if selected_score is None:
            return False
        try:
            threshold = float(self.settings.get("problem_auto_accept_logprob", -0.30))
            return float(selected_score) >= threshold
        except (TypeError, ValueError):
            return False

    def contact_confidence_is_high(self, selected_score):
        if selected_score is None:
            return False
        try:
            threshold = float(self.settings.get("contact_auto_accept_logprob", -0.30))
            return float(selected_score) >= threshold
        except (TypeError, ValueError):
            return False

    def can_auto_finalize(self, problem_score):
        return bool(
            self.has_complete_ticket_data()
            and self.company_trusted
            and self.contact_trusted
            and self.problem_confidence_is_high(problem_score)
        )

    async def finalize_ticket(self, uncertain=False, silent=False, warning_text=""):
        ticket = dict(self.ticket_data)
        if uncertain:
            warning = warning_text.strip() or (
                "UWAGA: Agent głosowy nie zdołał jednoznacznie potwierdzić danych po 3 próbach poprawki. "
                "Wymagany kontakt zwrotny z osobą zgłaszającą w celu doprecyzowania zgłoszenia."
            )
            existing_summary = str(ticket.get("summary", "") or "").strip()
            ticket["summary"] = (existing_summary + "\n" + warning).strip()
            ticket["uncertain_transcription"] = True
        try:
            token = decrypt_secret(self.settings.get("icp_token_enc", ""))
            client = ICProjectClient(
                self.settings.get("icp_instance", ""),
                token,
                self.settings.get("icp_board_column", ""),
            )
            try:
                add_message(
                    self.call_id,
                    "ticket",
                    json.dumps(
                        {
                            "company": str(ticket.get("company", "") or ""),
                            "contact": str(ticket.get("contact", "") or ""),
                            "description": str(ticket.get("description", "") or ""),
                            "uncertain": bool(uncertain),
                            "field_meta": {
                                field: dict(self.ticket_field_meta.get(field, {"source": "unknown"}))
                                for field in ("company", "contact", "description")
                            },
                        },
                        ensure_ascii=False,
                    ),
                )
            except Exception:
                log.warning("[%s] Could not persist ticket snapshot", self.call_id, exc_info=True)

            created = await client.create_task(ticket, self.settings.get("icp_priority", "normal"))
            ticket_no = created.get("number") or created.get("shortCode") or ""
            self.ticket_ref = str(ticket_no or created.get("id") or "")
            self.final_status = "completed_uncertain" if uncertain else "completed"
            if not silent:
                if uncertain and warning_text:
                    await self.say(
                        "Dziękuję. Zgłoszenie zostało zapisane. "
                        "Ktoś z serwisu skontaktuje się w tej sprawie. Do widzenia."
                    )
                elif uncertain:
                    await self.say(
                        "Dziękuję. Zgłoszenie zostało przyjęte. "
                        "Nie udało mi się dokładnie rozpoznać wszystkich poprawek. "
                        "Ktoś z serwisu skontaktuje się w celu doprecyzowania. Do widzenia."
                    )
                else:
                    await self.say(
                        "Dziękuję. Zgłoszenie zostało zapisane. "
                        "Ktoś z serwisu skontaktuje się w tej sprawie. Do widzenia."
                    )
            self.closed = True
            if not silent:
                await asyncio.sleep(0.3)
            self.writer.close()
            return True
        except Exception as e:
            self.final_status = "icp_error"
            self.final_error = str(e)
            log.exception("[%s] ICP create error", self.call_id)
            if not silent:
                await self.say(
                    "Nie udało się zapisać zgłoszenia w systemie. "
                    "Proszę skontaktować się z serwisem. Do widzenia."
                )
            self.closed = True
            if not silent:
                await asyncio.sleep(0.2)
            self.writer.close()
            return False

    async def interpret_fallback(self, expected: str, text: str):
        try:
            return await interpret_turn(
                self.settings["ollama_url"],
                self.settings["ollama_model"],
                expected,
                text,
                dict(self.ticket_data),
            )
        except Exception:
            log.exception("[%s] LLM fallback error", self.call_id)
            return {"intent": "unknown", "company": "", "contact": "", "description": ""}

    async def refuse_out_of_scope(self):
        if self.company_confirmation_pending:
            await self.say("Mogę obsłużyć tylko bieżące zgłoszenie. Proszę powiedzieć tak albo nie.")
        elif self.confirmation_pending:
            await self.say("Mogę obsłużyć tylko bieżące zgłoszenie. Proszę powiedzieć tak albo nie.")
        elif self.awaiting_correction:
            if self.correction_field == "company":
                await self.say("Mogę obsłużyć tylko bieżące zgłoszenie. Proszę podać poprawną nazwę firmy.")
            elif self.correction_field == "contact":
                await self.say("Mogę obsłużyć tylko bieżące zgłoszenie. Proszę podać poprawny numer telefonu.")
            elif self.correction_field == "description":
                await self.say("Mogę obsłużyć tylko bieżące zgłoszenie. Proszę podać poprawny opis problemu.")
            else:
                await self.say(
                    "Mogę obsłużyć tylko bieżące zgłoszenie. "
                    "Proszę wskazać: nazwa firmy, numer kontaktowy albo opis problemu."
                )
        elif self.awaiting_company:
            await self.say("Mogę obsłużyć tylko bieżące zgłoszenie. Proszę podać nazwę firmy.")
        elif self.awaiting_contact:
            await self.say("Mogę obsłużyć tylko bieżące zgłoszenie. Proszę podać numer telefonu kontaktowego.")
        elif self.awaiting_problem:
            await self.say("Mogę obsłużyć tylko bieżące zgłoszenie. Proszę opisać problem.")
        else:
            await self.say("Mogę pomóc wyłącznie w rejestracji bieżącego zgłoszenia serwisowego.")

    async def start(self):
        company = str(self.ticket_data.get("company", "") or "").strip()
        contact = str(self.ticket_data.get("contact", "") or "").strip()

        if company and contact:
            self.awaiting_problem = True
            await self.say(
                f"Dzień dobry. Tu automatyczny asystent PECEMED. "
                f"Rozpoznaję numer jako {company}. Proszę opisać problem."
            )
        elif contact:
            self.awaiting_company = True
            await self.say(
                "Dzień dobry. Tu automatyczny asystent PECEMED. "
                "Numer kontaktowy został rozpoznany automatycznie. "
                "Proszę podać nazwę firmy."
            )
        elif company:
            self.awaiting_contact = True
            await self.say(
                f"Dzień dobry. Tu automatyczny asystent PECEMED. "
                f"Firma: {company}. Proszę podać numer telefonu kontaktowego."
            )
        else:
            self.awaiting_company = True
            await self.say(
                "Dzień dobry. Tu automatyczny asystent PECEMED. "
                "Proszę podać nazwę firmy."
            )

    async def handle_pcm(self, payload: bytes):
        if time.monotonic() < self.listen_not_before:
            return
        self.frame_buf.extend(payload)
        while len(self.frame_buf) >= 320:
            frame = bytes(self.frame_buf[:320])
            del self.frame_buf[:320]
            self.pre_roll.append(frame)
            try:
                is_speech = self.vad.is_speech(frame, 8000)
            except Exception:
                is_speech = False

            if is_speech:
                if not self.speaking:
                    self.speaking = True
                    self.speech.extend(b"".join(self.pre_roll))
                self.speech.extend(frame)
                self.silence_frames = 0
            elif self.speaking:
                self.speech.extend(frame)
                self.silence_frames += 1
                silence_ms = self.silence_frames * 20
                silence_target_ms = 350 if (self.confirmation_pending or self.company_confirmation_pending) else int(self.settings.get("silence_ms", 900))
                if silence_ms >= silence_target_ms:
                    pcm = bytes(self.speech)
                    self.speech.clear()
                    self.speaking = False
                    self.silence_frames = 0
                    if len(pcm) >= 320 * 15:
                        await self.process_utterance(pcm)

    def stt_mode_for_state(self):
        if self.confirmation_pending or self.company_confirmation_pending:
            return "confirmation"
        if self.awaiting_correction and self.correction_field == "company":
            return "company"
        if self.awaiting_company:
            return "company"
        if self.awaiting_correction and self.correction_field == "contact":
            return "contact"
        if self.awaiting_contact:
            return "contact"
        if self.awaiting_correction and self.correction_field == "description":
            return "problem"
        if self.awaiting_problem:
            return "problem"
        return "normal"

    def stt_prompt_for_state(self):
        customer_names = ", ".join(x["name"] for x in self.customer_directory)
        base = (self.settings.get("stt_prompt", "") or "").strip()

        if self.confirmation_pending or self.company_confirmation_pending:
            # Keep this deliberately tiny. Longer prompts can be hallucinated
            # verbatim by Whisper on very short telephone utterances.
            return "tak, nie"

        if self.awaiting_correction and self.correction_field == "company":
            return "Dzwoniący podaje nazwę swojej firmy po polsku."

        if self.awaiting_company:
            return "Dzwoniący podaje nazwę swojej firmy po polsku."

        if self.awaiting_correction and self.correction_field == "contact":
            return "Numer telefonu. Cyfry od zera do dziewięciu."

        if self.awaiting_contact:
            return "Numer telefonu. Cyfry od zera do dziewięciu."

        if self.awaiting_correction and self.correction_field == "description":
            hint = (self.settings.get("stt_problem_hint", "") or "").strip()
            return ("Dzwoniący opisuje problem techniczny lub usterkę po polsku. " + hint).strip()

        if self.awaiting_problem:
            hint = (self.settings.get("stt_problem_hint", "") or "").strip()
            return ("Dzwoniący opisuje problem techniczny lub usterkę po polsku. " + hint).strip()

        prompt = base
        if customer_names:
            prompt = (prompt + " Nazwy klientów: " + customer_names).strip()
        return prompt

    async def handle_dtmf(self, payload: bytes):
        if not self.awaiting_contact_dtmf:
            return
        try:
            chars = payload.decode("ascii", errors="ignore")
        except Exception:
            chars = ""
        for ch in chars:
            if ch.isdigit():
                if len(self.dtmf_contact_buffer) < 15:
                    self.dtmf_contact_buffer += ch
            elif ch == "#":
                phone = extract_phone_digits(self.dtmf_contact_buffer, self.settings.get("phone_validation_mode", "pl"))
                if not phone:
                    self.dtmf_contact_buffer = ""
                    await self.say("Numer ma nieprawidłową długość. Proszę wpisać dziewięć cyfr i zakończyć krzyżykiem.")
                    return
                context = self.dtmf_contact_context
                self.awaiting_contact_dtmf = False
                self.dtmf_contact_buffer = ""
                self.dtmf_contact_context = ""
                self.contact_attempts = 0
                self.ticket_data["contact"] = phone
                self.contact_trusted = True
                self.mark_ticket_field("contact", "dtmf", None, True)
                matched_customer, match_score = match_customer(str(self.ticket_data.get("company", "") or ""), phone, self.customer_directory)
                self.record_customer_match("dtmf_contact", str(self.ticket_data.get("company", "") or ""), phone, matched_customer, match_score)
                if matched_customer:
                    self.ticket_data["company"] = matched_customer["name"]
                    self.company_trusted = True
                    self.mark_ticket_field("company", "directory", match_score, True)
                    if matched_customer.get("phone"):
                        self.ticket_data["contact"] = matched_customer["phone"]
                        self.contact_trusted = True
                        self.mark_ticket_field("contact", "directory", None, True)
                if context == "correction":
                    self.awaiting_correction = False
                    self.correction_field = ""
                    self.correction_attempts += 1
                    self.confirmation_pending = True
                    await self.say_confirmation_summary()
                    return
                self.awaiting_contact = False
                if self.ticket_data.get("description"):
                    if self.can_auto_finalize(self.early_problem_score):
                        await self.finalize_ticket()
                    else:
                        self.confirmation_pending = True
                        await self.say_confirmation_summary()
                    return
                self.awaiting_problem = True
                await self.say("Dziękuję. Proszę opisać problem.")
                return

    async def process_utterance(self, pcm: bytes):
        self.turns += 1
        try:
            stt_result = await asyncio.to_thread(
                transcribe_pcm16,
                pcm,
                self.settings["whisper_model"],
                self.settings["whisper_device"],
                self.settings["whisper_compute_type"],
                8000,
                self.stt_prompt_for_state(),
                self.stt_mode_for_state(),
                True,
                int(self.settings.get("stt_workers", 2) or 2),
                self.call_id,
            )
            if isinstance(stt_result, dict):
                text = str(stt_result.get("selected", "") or "")
                selected_score = stt_result.get("selected_score")
                add_message(
                    self.call_id,
                    "stt_debug",
                    json.dumps(stt_result, ensure_ascii=False),
                )
            else:
                # Backward-compatible fallback for patched tests / older tools.
                text = str(stt_result or "")
                selected_score = None
        except Exception:
            log.exception("[%s] STT error", self.call_id)
            await self.say("Nie udało mi się rozpoznać wypowiedzi. Proszę powtórzyć.")
            return

        if not text:
            self.stt_misses += 1
            now = time.monotonic()
            audio_seconds = len(pcm) / 16000.0
            retry_reason = stt_result.get("retry_reason", "") if isinstance(stt_result, dict) else ""
            # Silence is allowed only for a genuinely short residual prompt-leak
            # immediately after TTS. Never let a stale/misclassified retry_reason
            # suppress a real 1-2 second caller utterance.
            if retry_reason == "short_rejected_audio" and audio_seconds <= 0.80:
                log.info(
                    "[%s] Rejected short residual audio silently (%.2fs, reason=%s)",
                    self.call_id,
                    audio_seconds,
                    retry_reason,
                )
                return
            if audio_seconds >= 0.50:
                self.stt_misses = 0
                self.last_repeat_prompt = now
                log.info(
                    "[%s] Rejected real utterance; asking for repeat (%.2fs, reason=%s)",
                    self.call_id,
                    audio_seconds,
                    retry_reason or "empty",
                )
                await self.say("Nie dosłyszałem. Proszę powtórzyć.")
            return

        self.stt_misses = 0

        log.info("[%s] STT: %s", self.call_id, text)
        add_message(self.call_id, "user", text)
        self.history.append({"role": "user", "content": text})

        normalized = " ".join(text.lower().strip(" .,!?:;").split())

        # An explicit "do not create / cancel the ticket" always wins over a
        # problem description. This must be checked before any state can save.
        if looks_like_ticket_cancellation(text):
            self.final_status = "caller_cancelled"
            self.confirmation_pending = False
            self.company_confirmation_pending = False
            self.awaiting_correction = False
            self.awaiting_company = False
            self.awaiting_contact = False
            self.awaiting_problem = False
            await self.say("Rozumiem. Nie będę zakładać zgłoszenia. Do widzenia.")
            self.closed = True
            try:
                self.writer.close()
            except Exception:
                pass
            return

        # Explicit hostile/dismissive phrases mean the caller is ending the
        # interaction, not describing a service problem.
        if looks_like_abusive_dismissal(text):
            self.final_status = "caller_ended"
            await self.say("Rozumiem. Kończę rozmowę. Do widzenia.")
            self.closed = True
            try:
                self.writer.close()
            except Exception:
                pass
            return

        # Goodbye is global conversation intent. Handle it before any local
        # yes/no state (including company confirmation), otherwise "do widzenia"
        # can be mistaken for an invalid confirmation response.
        goodbye_phrases = (
            "do widzenia",
            "dziękuję do widzenia",
            "dziekuje do widzenia",
            "to wszystko",
            "koniec",
        )
        if any(phrase in normalized for phrase in goodbye_phrases):
            # Clear local confirmation states before ending the call so the
            # session cannot remain logically pending after a goodbye.
            self.company_confirmation_pending = False
            self.confirmation_pending = False
            if self.has_complete_ticket_data() and not self.ticket_ref:
                await self.finalize_ticket(
                    uncertain=True,
                    warning_text=(
                        "UWAGA: Rozmówca zakończył rozmowę po podaniu opisu problemu, "
                        "bez końcowego potwierdzenia danych."
                    ),
                )
                return
            self.final_status = "caller_ended"
            await self.say("Dziękuję za rozmowę. Do widzenia.")
            self.closed = True
            await asyncio.sleep(0.2)
            self.writer.close()
            return

        # Treat obvious attempts to alter the agent's rules as untrusted content.
        # They never reach field assignment, confirmation or the general LLM path.
        if looks_like_prompt_injection(text):
            log.warning("[%s] Blocked prompt-injection-like utterance: %s", self.call_id, text[:300])
            await self.refuse_out_of_scope()
            return

        if self.company_confirmation_pending:
            yes_phrases = (
                "tak", "tak zgadza się", "tak zgadza sie", "zgadza się", "zgadza sie",
                "potwierdzam", "dobrze", "tak dobrze",
            )
            no_phrases = (
                "nie", "nie zgadza się", "nie zgadza sie", "źle", "zle",
                "niepoprawne", "popraw", "zmień", "zmien",
            )

            if matches_confirmation_phrase(normalized, yes_phrases):
                self.ticket_data["company"] = self.company_candidate
                self.company_trusted = True
                self.mark_ticket_field("company", "confirmed_stt", self.company_candidate_score, True)
                if self.company_candidate_phone and not self.ticket_data.get("contact"):
                    self.ticket_data["contact"] = self.company_candidate_phone
                    self.contact_trusted = self.contact_confidence_is_high(self.company_candidate_score)
                    self.mark_ticket_field(
                        "contact",
                        "stt",
                        self.company_candidate_score,
                        self.contact_trusted,
                    )
                context = self.company_confirmation_context
                self.company_confirmation_pending = False
                self.company_candidate = ""
                self.company_candidate_score = None
                self.company_confirmation_context = ""
                self.company_candidate_phone = ""

                if context == "correction":
                    self.awaiting_correction = False
                    self.correction_field = ""
                    self.correction_attempts += 1
                    if self.correction_attempts >= 3:
                        await self.finalize_ticket(uncertain=True)
                        return
                    self.confirmation_pending = True
                    self.confirmation_misses = 0
                    await self.say_confirmation_summary()
                    return

                self.awaiting_company = False

                if self.ticket_data.get("description"):
                    if not self.ticket_data.get("title"):
                        description = str(self.ticket_data.get("description", "") or "").strip()
                        self.ticket_data["title"] = description[:80] or "Zgłoszenie telefoniczne"

                    if self.ticket_data.get("contact"):
                        self.awaiting_contact = False
                        self.awaiting_problem = False
                        if self.can_auto_finalize(self.early_problem_score):
                            await self.finalize_ticket()
                        else:
                            self.confirmation_pending = True
                            self.confirmation_misses = 0
                            await self.say_confirmation_summary()
                        return

                    self.awaiting_contact = True
                    self.awaiting_problem = False
                    await self.say("Dziękuję. Proszę podać numer telefonu kontaktowego.")
                    return

                if self.ticket_data.get("contact"):
                    self.awaiting_problem = True
                    await self.say("Dziękuję. Proszę opisać problem.")
                else:
                    self.awaiting_contact = True
                    await self.say("Dziękuję. Proszę podać numer telefonu kontaktowego.")
                return

            if matches_confirmation_phrase(normalized, no_phrases):
                context = self.company_confirmation_context
                rejected_candidate = self.company_candidate
                rejected_normalized = normalize_company(rejected_candidate)
                if rejected_normalized:
                    self.rejected_company_names.add(rejected_normalized)
                if (
                    rejected_normalized
                    and normalize_company(str(self.ticket_data.get("company", "") or "")) == rejected_normalized
                    and not self.company_trusted
                ):
                    self.ticket_data.pop("company", None)
                self.company_confirmation_pending = False
                self.company_candidate = ""
                self.company_candidate_score = None
                self.company_confirmation_context = ""
                self.company_candidate_phone = ""
                if context == "correction":
                    self.awaiting_correction = True
                    self.correction_field = "company"
                    await self.say("Dobrze. Proszę podać poprawną nazwę firmy jeszcze raz.")
                else:
                    self.awaiting_company = True
                    await self.say("Dobrze. Proszę podać samą nazwę firmy jeszcze raz.")
                return

            await self.say(
                "Nie rozpoznałem jednoznacznej odpowiedzi. "
                "Proszę powiedzieć tylko tak albo nie."
            )
            return

        if self.confirmation_pending:
            yes_phrases = (
                "tak",
                "tak zgadza się",
                "tak zgadza sie",
                "zgadza się",
                "zgadza sie",
                "potwierdzam",
                "wszystko się zgadza",
                "wszystko sie zgadza",
            )
            no_phrases = (
                "nie",
                "nie zgadza się",
                "nie zgadza sie",
                "niepoprawne",
                "błąd",
                "blad",
                "popraw",
                "zmień",
                "zmien",
            )

            if matches_confirmation_phrase(normalized, yes_phrases):
                self.confirmation_pending = False
                self.confirmation_misses = 0
                await self.finalize_ticket()
                return

            if matches_confirmation_phrase(normalized, no_phrases):
                self.confirmation_pending = False
                self.confirmation_misses = 0
                self.awaiting_correction = True
                self.awaiting_company = False
                self.awaiting_contact = False
                self.awaiting_problem = False

                # When CallerID was matched to the customer directory, company
                # and contact are trusted data. Only the spoken problem can
                # require correction.
                if self.caller_matched_customer:
                    self.correction_field = "description"
                    await self.say("Dobrze. Proszę podać poprawny opis problemu.")
                else:
                    self.correction_field = ""
                    await self.say(
                        "Dobrze. Proszę podać ponownie tylko dane, które mam poprawić: "
                        "nazwę firmy, numer kontaktowy albo opis problemu."
                    )
                return

            # Ambiguous confirmation must never create a ticket.
            # LLM may help detect a correction/negative intent, but it is not
            # allowed to turn an unclear transcript into a positive confirmation.
            interpreted = await self.interpret_fallback("potwierdzenie danych tak/nie", text)
            if interpreted.get("blocked"):
                await self.refuse_out_of_scope()
                return
            intent = str(interpreted.get("intent", "")).lower()

            if intent in ("confirm_no", "correction"):
                self.confirmation_pending = False
                self.confirmation_misses = 0
                self.awaiting_correction = True
                self.awaiting_company = False
                self.awaiting_contact = False
                self.awaiting_problem = False
                if self.caller_matched_customer:
                    self.correction_field = "description"
                    await self.say("Dobrze. Proszę podać poprawny opis problemu.")
                else:
                    self.correction_field = ""
                    await self.say(
                        "Dobrze. Proszę podać tylko dane, które mam poprawić: "
                        "nazwę firmy, numer kontaktowy albo opis problemu."
                    )
                return

            self.confirmation_misses += 1
            if self.confirmation_misses >= 1:
                self.confirmation_misses = 0
                await self.say(
                    "Nie rozpoznałem jednoznacznej odpowiedzi. "
                    "Proszę powiedzieć tylko tak albo nie."
                )
            return

        # Two-step correction flow:
        # 1) caller selects which field is wrong,
        # 2) next utterance becomes the replacement value.
        if self.awaiting_correction:
            if not self.correction_field:
                if any(x in normalized for x in ("firma", "nazwa firmy", "nazwa klienta")):
                    self.correction_field = "company"
                    await self.say("Proszę podać poprawną nazwę firmy.")
                    return

                if any(x in normalized for x in ("numer", "telefon", "kontakt", "numer kontaktowy")):
                    self.correction_field = "contact"
                    await self.say("Proszę podać poprawny numer telefonu kontaktowego.")
                    return

                if any(x in normalized for x in ("opis", "opis problemu", "problem")):
                    self.correction_field = "description"
                    await self.say("Proszę podać poprawny opis problemu.")
                    return

                interpreted = await self.interpret_fallback(
                    "wybór pola do poprawy: firma, numer kontaktowy albo opis problemu",
                    text,
                )
                if interpreted.get("blocked"):
                    await self.refuse_out_of_scope()
                    return
                intent = str(interpreted.get("intent", "")).lower()
                if interpreted.get("company"):
                    self.correction_field = "company"
                    await self.say("Proszę podać poprawną nazwę firmy.")
                    return
                if interpreted.get("contact"):
                    self.correction_field = "contact"
                    await self.say("Proszę podać poprawny numer telefonu kontaktowego.")
                    return
                if interpreted.get("description") or intent == "problem":
                    self.correction_field = "description"
                    await self.say("Proszę podać poprawny opis problemu.")
                    return

                await self.say(
                    "Proszę powiedzieć, co mam poprawić: nazwę firmy, numer kontaktowy albo opis problemu."
                )
                return

            if self.correction_field == "company":
                company_text = clean_company_display_name(text) or text.strip()
                if looks_like_invalid_company_name(company_text):
                    await self.say(
                        "Nie udało mi się wiarygodnie rozpoznać nazwy firmy. "
                        "Proszę podać ją ponownie, możliwie krótko i wyraźnie."
                    )
                    return
                matched_customer, match_score = match_customer(company_text, "", self.customer_directory)
                self.record_customer_match("correction_company", company_text, "", matched_customer, match_score)
                if matched_customer:
                    self.ticket_data["company"] = matched_customer["name"]
                    self.company_trusted = True
                    self.mark_ticket_field("company", "directory", match_score, True)
                else:
                    confirm_threshold = float(self.settings.get("company_confirm_logprob", -0.55))
                    low_confidence = (
                        selected_score is not None
                        and float(selected_score) < confirm_threshold
                    )
                    if low_confidence:
                        self.company_confirmation_pending = True
                        self.company_candidate = company_text
                        self.company_candidate_score = float(selected_score)
                        self.company_confirmation_context = "correction"
                        self.company_candidate_phone = ""
                        await self.say(
                            f"Czy dobrze zrozumiałem poprawioną nazwę: {company_text}? "
                            "Proszę powiedzieć tak albo nie."
                        )
                        return
                    self.ticket_data["company"] = company_text
                    self.company_trusted = bool(
                        selected_score is not None
                        and float(selected_score) >= confirm_threshold
                    )
                    self.mark_ticket_field("company", "stt", selected_score, self.company_trusted)

            elif self.correction_field == "contact":
                phone = extract_phone_digits(text, self.settings.get("phone_validation_mode", "pl"))
                if not phone:
                    self.contact_attempts += 1
                    if self.contact_attempts >= 2:
                        self.awaiting_contact_dtmf = True
                        self.dtmf_contact_buffer = ""
                        self.dtmf_contact_context = "correction"
                        await self.say("Proszę wpisać dziewięć cyfr na klawiaturze telefonu i zakończyć krzyżykiem.")
                    else:
                        await self.say("Nie udało mi się rozpoznać numeru. Proszę podać go cyfra po cyfrze.")
                    return
                self.contact_attempts = 0
                self.awaiting_contact_dtmf = False
                self.ticket_data["contact"] = phone
                self.contact_trusted = self.contact_confidence_is_high(selected_score)
                self.mark_ticket_field("contact", "stt", selected_score, self.contact_trusted)
                matched_customer, match_score = match_customer(
                    str(self.ticket_data.get("company", "") or ""),
                    phone,
                    self.customer_directory,
                )
                self.record_customer_match(
                    "correction_contact",
                    str(self.ticket_data.get("company", "") or ""),
                    phone,
                    matched_customer,
                    match_score,
                )
                if matched_customer:
                    self.ticket_data["company"] = matched_customer["name"]
                    self.company_trusted = True
                    self.mark_ticket_field("company", "directory", match_score, True)
                    if matched_customer.get("phone"):
                        self.ticket_data["contact"] = matched_customer["phone"]
                        self.contact_trusted = True
                        self.mark_ticket_field("contact", "directory", None, True)

            elif self.correction_field == "description":
                if looks_like_human_handoff_request(text):
                    await self.say(
                        "Mogę przyjąć zgłoszenie dla serwisu. "
                        "Proszę opisać problem lub usterkę, a zgłoszenie przekażę do obsługi."
                    )
                    return
                if looks_like_ticket_meta_request(text):
                    await self.say(
                        "To brzmi jak polecenie dotyczące zgłoszenia. "
                        "Proszę opisać, na czym polega problem lub usterka."
                    )
                    return
                self.ticket_data["description"] = text.strip()
                self.mark_ticket_field("description", "stt", selected_score, self.problem_confidence_is_high(selected_score))
                self.ticket_data["title"] = text.strip()[:80] or "Zgłoszenie telefoniczne"

                if self.can_auto_finalize(selected_score):
                    self.awaiting_correction = False
                    self.correction_field = ""
                    self.confirmation_pending = False
                    await self.finalize_ticket()
                    return

            self.awaiting_correction = False
            self.correction_field = ""
            self.correction_attempts += 1

            if self.correction_attempts >= 3:
                self.confirmation_pending = False
                self.confirmation_misses = 0
                await self.finalize_ticket(uncertain=True)
                return

            self.confirmation_pending = True
            self.confirmation_misses = 0
            await self.say_confirmation_summary()
            return

        # Fast deterministic state machine for normal calls.
        # Ollama is only a fallback for corrections / unusual utterances.
        if self.awaiting_company:
            phone = extract_phone_digits(text, self.settings.get("phone_validation_mode", "pl"))
            company_text = clean_company_display_name(text) or text.strip()

            if looks_like_invalid_company_name(company_text):
                await self.say(
                    "Nie udało mi się wiarygodnie rozpoznać nazwy firmy. "
                    "Proszę podać samą nazwę firmy jeszcze raz."
                )
                return

            looks_like_problem = any(
                phrase in normalized
                for phrase in ("problem", "nie działa", "nie dziala", "awaria", "błąd", "blad", "usterka", "nie mogę", "nie moge")
            )
            if looks_like_problem:
                deterministic_problem = extract_problem_fragment(text)
                if deterministic_problem and not self.ticket_data.get("description"):
                    self.ticket_data["description"] = deterministic_problem
                    self.early_problem_score = selected_score
                    self.mark_ticket_field(
                        "description",
                        "stt",
                        selected_score,
                        self.problem_confidence_is_high(selected_score),
                    )

                interpreted = await self.interpret_fallback("nazwa firmy", text)
                if interpreted.get("blocked"):
                    await self.refuse_out_of_scope()
                    return
                if interpreted.get("company"):
                    company_text = clean_company_display_name(str(interpreted["company"])) or str(interpreted["company"]).strip()
                if interpreted.get("contact") and not phone:
                    phone = extract_phone_digits(
                        str(interpreted["contact"]),
                        self.settings.get("phone_validation_mode", "pl"),
                    )
                if interpreted.get("description") and not self.ticket_data.get("description"):
                    self.ticket_data["description"] = str(interpreted["description"]).strip()
                    self.early_problem_score = selected_score

            matched_customer, match_score = match_customer(
                company_text,
                phone,
                self.customer_directory,
            )
            self.record_customer_match("initial_company", company_text, phone, matched_customer, match_score)
            if matched_customer:
                self.ticket_data["company"] = matched_customer["name"]
                self.company_trusted = True
                self.mark_ticket_field("company", "directory", match_score, True)
                if matched_customer.get("phone"):
                    self.ticket_data["contact"] = matched_customer["phone"]
                    self.contact_trusted = True
                    self.mark_ticket_field("contact", "directory", None, True)
            else:
                confirm_threshold = float(self.settings.get("company_confirm_logprob", -0.55))
                low_confidence = (
                    selected_score is not None
                    and float(selected_score) < confirm_threshold
                )
                was_rejected = normalize_company(company_text) in self.rejected_company_names
                if low_confidence or was_rejected:
                    self.company_confirmation_pending = True
                    self.company_candidate = company_text
                    self.company_candidate_score = float(selected_score)
                    self.company_confirmation_context = "initial"
                    self.company_candidate_phone = phone or ""
                    self.awaiting_company = False
                    await self.say(f"Czy dobrze zrozumiałem: firma {company_text}? Proszę powiedzieć tak albo nie.")
                    return
                self.ticket_data["company"] = company_text
                self.company_trusted = bool(
                    selected_score is not None
                    and float(selected_score) >= confirm_threshold
                )
                self.mark_ticket_field("company", "stt", selected_score, self.company_trusted)

            if phone and not self.ticket_data.get("contact"):
                self.ticket_data["contact"] = phone
                self.contact_trusted = self.contact_confidence_is_high(selected_score)
                self.mark_ticket_field("contact", "stt", selected_score, self.contact_trusted)

            self.awaiting_company = False

            if self.ticket_data.get("description"):
                if not self.ticket_data.get("title"):
                    description = str(self.ticket_data.get("description", "") or "").strip()
                    self.ticket_data["title"] = description[:80] or "Zgłoszenie telefoniczne"

                if self.ticket_data.get("contact"):
                    self.awaiting_contact = False
                    self.awaiting_problem = False
                    if self.can_auto_finalize(self.early_problem_score):
                        await self.finalize_ticket()
                    else:
                        self.confirmation_pending = True
                        self.confirmation_misses = 0
                        await self.say_confirmation_summary()
                    return

                self.awaiting_contact = True
                self.awaiting_problem = False
                await self.say("Dziękuję. Proszę podać numer telefonu kontaktowego.")
                return

            if self.ticket_data.get("contact"):
                self.awaiting_problem = True
                await self.say("Dziękuję. Proszę opisać problem.")
            else:
                self.awaiting_contact = True
                await self.say("Dziękuję. Proszę podać numer telefonu kontaktowego.")
            return

        if self.awaiting_contact:
            phone = extract_phone_digits(text, self.settings.get("phone_validation_mode", "pl"))
            if not phone:
                    self.contact_attempts += 1
                    if self.contact_attempts >= 2:
                        self.awaiting_contact_dtmf = True
                        self.dtmf_contact_buffer = ""
                        self.dtmf_contact_context = "initial"
                        await self.say(
                            "Nie udało mi się pewnie rozpoznać numeru. "
                            "Proszę wpisać dziewięć cyfr na klawiaturze telefonu i zakończyć krzyżykiem."
                        )
                    else:
                        await self.say(
                            "Nie udało mi się rozpoznać numeru telefonu. "
                            "Proszę podać go cyfra po cyfrze."
                        )
                    return

            self.contact_attempts = 0
            self.awaiting_contact_dtmf = False
            self.ticket_data["contact"] = phone
            self.contact_trusted = self.contact_confidence_is_high(selected_score)
            self.mark_ticket_field("contact", "stt", selected_score, self.contact_trusted)

            matched_customer, match_score = match_customer(
                str(self.ticket_data.get("company", "") or ""),
                phone,
                self.customer_directory,
            )
            self.record_customer_match(
                "initial_contact",
                str(self.ticket_data.get("company", "") or ""),
                phone,
                matched_customer,
                match_score,
            )
            if matched_customer:
                self.ticket_data["company"] = matched_customer["name"]
                self.company_trusted = True
                self.mark_ticket_field("company", "directory", match_score, True)
                if matched_customer.get("phone"):
                    self.ticket_data["contact"] = matched_customer["phone"]
                    self.contact_trusted = True
                    self.mark_ticket_field("contact", "directory", None, True)

            self.awaiting_contact = False

            if self.ticket_data.get("description"):
                if not self.ticket_data.get("title"):
                    description = str(self.ticket_data.get("description", "") or "").strip()
                    self.ticket_data["title"] = description[:80] or "Zgłoszenie telefoniczne"

                self.awaiting_problem = False
                if self.can_auto_finalize(self.early_problem_score):
                    await self.finalize_ticket()
                else:
                    self.confirmation_pending = True
                    self.confirmation_misses = 0
                    await self.say_confirmation_summary()
                return

            self.awaiting_problem = True
            await self.say("Dziękuję. Proszę opisać problem.")
            return

        # Explicit conversation state beats LLM inference. If we just asked
        # for the problem, accept the next non-empty utterance as description.
        if self.awaiting_problem and not self.ticket_data.get("description"):
            if looks_like_human_handoff_request(text):
                await self.say(
                    "Mogę przyjąć zgłoszenie dla serwisu. "
                    "Proszę opisać problem lub usterkę, a zgłoszenie przekażę do obsługi."
                )
                return

            if looks_like_ticket_meta_request(text):
                await self.say(
                    "Oczywiście mogę zarejestrować zgłoszenie. "
                    "Proszę opisać, na czym polega problem, który mam przekazać do serwisu."
                )
                return

            self.ticket_data["description"] = text.strip()
            self.early_problem_score = selected_score
            self.mark_ticket_field("description", "stt", selected_score, self.problem_confidence_is_high(selected_score))
            self.awaiting_problem = False

            # Fast path for recognized callers: company and contact already came
            # from CallerID/customer directory, so an LLM call adds latency but
            # no useful information. Go straight to confirmation.
            company = str(self.ticket_data.get("company", "") or "").strip()
            contact = str(self.ticket_data.get("contact", "") or "").strip()
            description = str(self.ticket_data.get("description", "") or "").strip()
            if company and contact and description:
                if not self.ticket_data.get("title"):
                    self.ticket_data["title"] = description[:80] or "Zgłoszenie telefoniczne"

                self.awaiting_correction = False
                self.awaiting_company = False
                self.awaiting_contact = False
                self.awaiting_problem = False

                if self.can_auto_finalize(selected_score):
                    self.confirmation_pending = False
                    self.confirmation_misses = 0
                    await self.finalize_ticket()
                    return

                self.confirmation_pending = True
                self.confirmation_misses = 0
                await self.say_confirmation_summary()
                return

        # Give the LLM an explicit snapshot of already collected data. This is
        # more reliable with small local models than expecting them to reconstruct
        # state only from previous JSON turns.
        state_context = dict(self.ticket_data)
        llm_history = list(self.history)
        if state_context:
            llm_history.append({
                "role": "system",
                "content": "Już zebrane dane zgłoszenia (zachowaj je): "
                           + json.dumps(state_context, ensure_ascii=False),
            })

        try:
            result = await ask_ollama(
                self.settings["ollama_url"],
                self.settings["ollama_model"],
                self.settings["system_prompt"],
                llm_history,
            )
        except Exception:
            log.exception("[%s] LLM error", self.call_id)
            await self.say("Wystąpił chwilowy problem z systemem. Proszę spróbować ponownie.")
            return

        # LLM output is used only as a parser. The model is not allowed to
        # choose arbitrary customer-facing speech.
        reply = ""
        ticket_update = result.get("ticket") or {}

        # General LLM fallback is fill-only. It may add a missing field but it
        # may NEVER overwrite data already collected for this call.
        before_fill = {
            key: str(self.ticket_data.get(key, "") or "")
            for key in ("company", "contact", "description")
        }
        apply_llm_fill_only(self.ticket_data, ticket_update)
        for field in ("company", "contact", "description"):
            if not before_fill[field] and str(self.ticket_data.get(field, "") or "").strip():
                self.mark_ticket_field(field, "llm_fallback", None, False)

        # Normalize contact phone numbers recognized with spaces, commas or dashes.
        contact_value = str(self.ticket_data.get("contact", "") or "")
        contact_digits = extract_phone_digits(
            contact_value,
            self.settings.get("phone_validation_mode", "pl"),
        )
        if contact_digits:
            self.ticket_data["contact"] = contact_digits
        elif contact_value:
            self.ticket_data["contact"] = ""

        matched_customer, match_score = match_customer(
            str(self.ticket_data.get("company", "") or ""),
            str(self.ticket_data.get("contact", "") or ""),
            self.customer_directory,
        )
        self.record_customer_match(
            "fallback",
            str(self.ticket_data.get("company", "") or ""),
            str(self.ticket_data.get("contact", "") or ""),
            matched_customer,
            match_score,
        )
        if matched_customer:
            self.ticket_data["company"] = matched_customer["name"]
            self.company_trusted = True
            self.mark_ticket_field("company", "directory", match_score, True)
            if not self.ticket_data.get("contact") and matched_customer.get("phone"):
                self.ticket_data["contact"] = matched_customer["phone"]
                self.contact_trusted = True
                self.mark_ticket_field("contact", "directory", None, True)

        # If we are clearly asking for a problem and the caller gives a real
        # utterance, accept it as the description even if the LLM is too strict.
        previous_agent = ""
        for item in reversed(self.history[:-1]):
            if item.get("role") == "assistant":
                previous_agent = item.get("content", "")
                break
        problem_words = ("problem", "opis", "co się dzieje", "usterk")
        if (
            not self.ticket_data.get("description")
            and len(text.strip()) >= 3
            and not looks_like_ticket_meta_request(text)
            and any(word in previous_agent.lower() for word in problem_words)
        ):
            self.ticket_data["description"] = text.strip()

        if self.ticket_data.get("description") and not self.ticket_data.get("title"):
            desc = self.ticket_data["description"].strip()
            self.ticket_data["title"] = desc[:80] or "Zgłoszenie telefoniczne"

        result["ticket"] = dict(self.ticket_data)

        company_ok = bool(str(self.ticket_data.get("company", "")).strip())
        contact_ok = bool(str(self.ticket_data.get("contact", "")).strip())
        description_ok = bool(str(self.ticket_data.get("description", "")).strip())

        # Backend owns the conversation state and every spoken response.
        # The LLM cannot move the call to an unrelated topic.
        if company_ok and contact_ok and description_ok:
            result["done"] = True
            reply = "Dziękuję, mam potrzebne informacje."
        else:
            result["done"] = False
            self.awaiting_company = False
            self.awaiting_contact = False
            self.awaiting_problem = False

            if not company_ok:
                self.awaiting_company = True
                reply = "Proszę podać nazwę firmy."
            elif not contact_ok:
                self.awaiting_contact = True
                reply = "Proszę podać numer telefonu kontaktowego."
            elif not description_ok:
                self.awaiting_problem = True
                reply = "Proszę opisać problem."
            else:
                reply = "Proszę podać informacje dotyczące bieżącego zgłoszenia."

        self.history.append({"role": "assistant", "content": json.dumps(result, ensure_ascii=False)})

        if result.get("done"):
            company = str(self.ticket_data.get("company", "") or "").strip() or "nie podano"
            contact = str(self.ticket_data.get("contact", "") or "").strip() or "nie podano"
            spoken_contact = speak_phone(contact) if contact != "nie podano" else contact
            description = str(self.ticket_data.get("description", "") or "").strip() or "nie podano"

            self.awaiting_correction = False
            if self.can_auto_finalize(selected_score):
                self.confirmation_pending = False
                self.confirmation_misses = 0
                await self.finalize_ticket()
                return

            self.confirmation_pending = True
            self.confirmation_misses = 0
            await self.say_confirmation_summary()
            return

        if self.turns >= int(self.settings.get("max_turns", 8)):
            await self.say(
                "Nie udało się zebrać kompletu informacji. Proszę skontaktować się z serwisem."
            )
            self.final_status = "incomplete"
            self.closed = True
            self.writer.close()
            return

        reply_lower = reply.lower()
        if (
            not self.ticket_data.get("description")
            and any(word in reply_lower for word in ("opis problemu", "opisać problem", "opisz problem", "jaki jest problem"))
        ):
            self.awaiting_problem = True

        await self.say(reply)

async def handle_client(reader, writer):
    peer = writer.get_extra_info("peername")
    call_id = str(uuid.uuid4())
    session = CallSession(call_id, writer)
    log.info("AudioSocket connection from %s", peer)

    try:
        typ, payload = await read_packet(reader)
        if typ == TYPE_UUID and payload and len(payload) == 16:
            call_id = str(uuid.UUID(bytes=payload))
            session.call_id = call_id
            log.info("Call UUID: %s", call_id)
        else:
            log.warning("First packet was not UUID; continuing.")

        registered_caller = consume_caller(call_id)
        if registered_caller:
            caller_digits = re.sub(r"\D", "", registered_caller)
            if caller_digits:
                session.original_caller = caller_digits
                session.ticket_data["contact"] = caller_digits
                session.mark_ticket_field("contact", "callerid", None, True)
                matched_customer, match_score = match_customer(
                    "",
                    caller_digits,
                    session.customer_directory,
                )
                if matched_customer:
                    session.caller_matched_customer = True
                    session.company_trusted = True
                    session.contact_trusted = True
                    session.ticket_data["company"] = matched_customer["name"]
                    session.mark_ticket_field("company", "callerid_directory", match_score, True)
                    if matched_customer.get("phone"):
                        session.ticket_data["contact"] = matched_customer["phone"]
                        session.mark_ticket_field("contact", "callerid_directory", None, True)
                else:
                    # Preserve the actual inbound CallerID separately from any
                    # contact number later recognized or corrected by STT.
                    session.ticket_data["caller"] = caller_digits
                session.record_customer_match("callerid", "", caller_digits, matched_customer, match_score)
                log.info(
                    "[%s] CallerID registered: %s%s",
                    call_id,
                    session.ticket_data.get("contact", caller_digits),
                    f" -> {session.ticket_data.get('company')}" if session.ticket_data.get("company") else "",
                )

        call_started(call_id, peer)
        await session.start()

        while not session.closed:
            typ, payload = await read_packet(reader)
            if typ is None or typ == TYPE_HANGUP:
                break
            if typ == TYPE_PCM_8K:
                await session.handle_pcm(payload)
            elif typ == TYPE_DTMF:
                log.info("[%s] DTMF: %r", call_id, payload)
                await session.handle_dtmf(payload)
    except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
        log.info("[%s] AudioSocket peer closed connection", call_id)
    except RuntimeError as e:
        if "closed" in str(e).lower():
            log.info("[%s] AudioSocket transport already closed: %s", call_id, e)
        else:
            session.final_status = "error"
            session.final_error = str(e)
            log.exception("[%s] AudioSocket session error", call_id)
    except Exception as e:
        session.final_status = "error"
        session.final_error = str(e)
        log.exception("[%s] AudioSocket session error", call_id)
    finally:
        # If the caller hangs up after already providing all ticket data, save
        # the ticket even when the final yes/no confirmation never arrived.
        if (
            not session.ticket_ref
            and session.has_complete_ticket_data()
            and session.final_status not in ("completed", "completed_uncertain", "icp_error")
        ):
            await session.finalize_ticket(
                uncertain=True,
                silent=True,
                warning_text=(
                    "UWAGA: Rozmówca rozłączył się po podaniu opisu problemu, "
                    "bez końcowego potwierdzenia danych."
                ),
            )

        finish_call(
            call_id,
            status=session.final_status,
            ticket_ref=session.ticket_ref,
            error=session.final_error,
        )
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass
        log.info("[%s] Call closed", call_id)

async def start_audiosocket_server():
    s = load_settings()
    server = await asyncio.start_server(
        handle_client,
        s.get("audiosocket_host", "0.0.0.0"),
        int(s.get("audiosocket_port", 9019)),
    )
    log.info("AudioSocket listening on %s", ", ".join(str(x.getsockname()) for x in server.sockets))
    async with server:
        await server.serve_forever()
