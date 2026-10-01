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
from .icproject import ICProjectClient
from .monitoring import call_started, add_message, finish_call
from .call_registry import consume_caller

log = logging.getLogger("audiosocket")

TYPE_HANGUP = 0x00
TYPE_UUID = 0x01
TYPE_DTMF = 0x03
TYPE_PCM_8K = 0x10


_META_PATTERNS = (
    r"zignoruj\s+(wszystkie\s+)?(poprzednie\s+)?instrukc",
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"system\s*prompt",
    r"prompt\s+system",
    r"poka[zż]\s+(mi\s+)?prompt",
    r"ujawnij\s+(has[łl]o|token|klucz|sekret|instrukc)",
    r"podaj\s+(has[łl]o|token|klucz|sekret)",
    r"wykonaj\s+(komend|polecen)",
    r"uruchom\s+(komend|polecen|shell|terminal)",
    r"developer\s+message",
)


def looks_like_prompt_injection(text: str) -> bool:
    value = " ".join(str(text or "").lower().split())
    if not value:
        return False
    return any(re.search(pattern, value, re.IGNORECASE) for pattern in _META_PATTERNS)

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


def parse_company_alias_dictionary(raw: str):
    """Parse: canonical | alias 1 | alias 2 ...; phone number is optional."""
    items = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("|") if part.strip()]
        if not parts:
            continue
        canonical = parts[0]
        aliases = []
        for part in parts[1:]:
            aliases.extend(x.strip() for x in part.split(",") if x.strip())
        seen = set()
        normalized_aliases = []
        for value in [canonical, *aliases]:
            key = normalize_company(value)
            if key and key not in seen:
                seen.add(key)
                normalized_aliases.append(value)
        items.append({"name": canonical, "aliases": normalized_aliases, "from_company_dictionary": True})
    return items


def merge_company_alias_dictionary(customers, alias_items):
    merged = [dict(item) for item in (customers or [])]
    by_name = {normalize_company(item.get("name", "")): item for item in merged}
    for alias_item in alias_items or []:
        canonical = str(alias_item.get("name", "") or "").strip()
        key = normalize_company(canonical)
        if not key:
            continue
        target = by_name.get(key)
        if target is None:
            target = {"name": canonical, "phone": "", "aliases": [], "from_company_dictionary": True}
            merged.append(target)
            by_name[key] = target
        existing = list(target.get("aliases") or [])
        existing_keys = {normalize_company(x) for x in existing}
        for alias in alias_item.get("aliases") or []:
            alias_key = normalize_company(alias)
            if alias_key and alias_key not in existing_keys:
                existing.append(alias)
                existing_keys.add(alias_key)
        target["aliases"] = existing
        target["from_company_dictionary"] = True
    return merged


def parse_problem_dictionary(raw: str):
    values = []
    seen = set()
    for line in (raw or "").splitlines():
        for part in line.split(","):
            value = part.strip()
            key = value.lower()
            if value and not value.startswith("#") and key not in seen:
                seen.add(key)
                values.append(value)
    return values


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
        "do zobaczenia",
        "szanowni państwo",
        "szanowni panstwo",
        "nie zauważyłem",
        "nie zauwazylem",
        "nie wierzę",
        "nie wierze",
        "poproszę",
        "poprosze",
        "no dobra",
        "mamy to",
        "part ii",
        "nie wiem",
        "nie wiem jak się nazywa",
        "nie wiem jak sie nazywa",
        "wszystko w porządku",
        "wszystko w porzadku",
        "mówiłem o nazwę firmy",
        "mowilem o nazwe firmy",
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
        "szanowny", "no", "to jest", "yyy", "yyy yyy", "hmm",
        "albo", "a tu", "a to",
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
        polite_or_speech_words = {
            "bardzo", "dziekuje", "dziękuję", "dzieki", "dzięki",
            "dzwonie", "dzwonię", "dzwoniacy", "dzwoniący", "dzwoncy",
            "dzien", "dzień", "dobry", "czesc", "cześć", "szanowny",
            "co", "to", "jest", "tak", "no", "a", "poprosze", "poproszę",
            "mamy", "part", "ii", "nie", "zauwazylem", "zauważyłem", "wierze", "wierzę",
        }
        if words and all(word in polite_or_speech_words for word in words):
            return True

    if "www." in normalized or "http://" in normalized or "https://" in normalized:
        return True

    # A transcript consisting only of a domain/address is not a company name.
    if re.fullmatch(r"[a-z0-9.-]+\.(?:pl|com|eu|org|net)(?:\s+[a-z0-9.-]+\.(?:pl|com|eu|org|net))*", normalized):
        return True

    return False


def looks_like_invalid_problem_description(value: str):
    normalized = " ".join((value or "").lower().strip(" .,!?:;").split())
    if not normalized:
        return True
    folded = normalized.translate(str.maketrans({
        "ą": "a", "ć": "c", "ę": "e", "ł": "l",
        "ń": "n", "ó": "o", "ś": "s", "ź": "z", "ż": "z",
    }))
    if folded.startswith(("nazywam sie ", "mam na imie ", "jestem ")):
        return True
    bad_fragments = (
        "dzwoniacy opisuje problem",
        "dzieki za ogladanie",
        "dziekuje za ogladanie",
        "dzieki za uwage",
        "arigatou gozaimasu",
        "wszystko w porzadku",
    )
    if any(fragment in folded for fragment in bad_fragments):
        return True
    words = re.findall(r"[a-z0-9]+", folded)
    polite = {
        "dzien", "dobry", "dziekuje", "dzieki", "bardzo", "czesc",
        "witam", "pozdrawiam", "do", "widzenia", "zobaczenia", "tak", "nie",
    }
    return bool(words) and all(word in polite for word in words)


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
    if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in patterns):
        return True

    # Tolerate common one-character STT corruptions seen in phone audio.
    folded = normalized.translate(str.maketrans({
        "ą": "a", "ć": "c", "ę": "e", "ł": "l",
        "ń": "n", "ó": "o", "ś": "s", "ź": "z", "ż": "z",
    }))
    fuzzy_ticket = bool(re.search(r"\b(?:z|s|b)?(?:g|k|b)?loszen\w*\b|\bzlozen\w*\b|\bzbloszen\w*\b", folded))
    cancel_action = bool(re.search(
        r"\b(?:nie\s+)(?:za[kg]lad\w*|zaglad\w*|zaklad\w*|tworz\w*|rejestr\w*|zapis\w*)\b",
        folded,
    )) or bool(re.search(r"\b(?:anul\w*|wycof\w*|rezygn\w*)\b", folded))
    no_longer_needed = "niepotrzebn" in folded or bool(re.search(r"\bnie\s+potrzebn", folded))
    already_works = bool(re.search(r"\bjuz\b.*\bdziala", folded)) or "zaczelo dzialac" in folded

    # Require two semantic signals to avoid cancelling a normal technical report.
    if fuzzy_ticket and (cancel_action or no_longer_needed):
        return True
    if already_works and no_longer_needed:
        return True
    if already_works and cancel_action:
        return True
    return False


def looks_like_possible_cancellation(text: str):
    normalized = " ".join((text or "").lower().strip(" .,!?:;").split())
    if not normalized:
        return False
    signals = (
        "niepotrzebn", "nie potrzebn", "już działa", "juz dziala",
        "zaczęło działać", "zaczelo dzialac", "nie zak", "nie zag",
        "anul", "wycof", "rezygn",
    )
    return any(signal in normalized for signal in signals)


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


def extract_company_fragment(text: str, directory: list | None = None):
    """Extract a likely company/name fragment from a mixed company+problem utterance."""
    source = " ".join((text or "").strip().split())
    if not source:
        return ""

    # Split directly at the first concrete problem signal. Using the full
    # extracted problem sentence would lose the company when both appear in
    # the same sentence ("Tu Rehabilitacja ETOS nie działa nam poczta").
    lower = source.lower()
    problem_signals = (
        "nie działa", "nie dziala", "nie mogę", "nie moge", "nie można", "nie mozna",
        "błąd", "blad", "awaria", "usterka", "problem z", "brak ", "wyskakuje",
        "zawiesza", "rozłącza", "rozlacza", "wolno działa", "wolno dziala",
        "nie otwiera", "nie drukuje", "nie loguje", "nie zapisuje", "nie wysyła",
        "nie wysyla", "przestał", "przestal", "zepsuł", "zepsul",
    )
    positions = [lower.find(signal) for signal in problem_signals if lower.find(signal) >= 0]
    prefix = source[:min(positions)] if positions else source

    prefix = re.sub(
        r"^\s*(?:dzień dobry|dzien dobry|dobry wieczór|dobry wieczor|cześć|czesc|witam)[,.:;\-\s]*",
        "",
        prefix,
        flags=re.IGNORECASE,
    )
    prefix = re.sub(r"\b(?:tu|z tej strony|firma|spółka|spolka)\b", " ", prefix, flags=re.IGNORECASE)
    prefix = re.sub(r"\b(?:jest|jesteśmy|jestesmy)\b", " ", prefix, flags=re.IGNORECASE)
    prefix = re.sub(r"\s+", " ", prefix).strip(" ,.;:-")

    if directory:
        tokens = re.findall(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż0-9]+", prefix)
        best = None
        best_score = 0.0
        # Try contiguous n-grams; mixed utterances often contain extra filler
        # before/after the company name that hurts matching of the whole prefix.
        for size in range(min(5, len(tokens)), 0, -1):
            for start in range(0, len(tokens) - size + 1):
                candidate = " ".join(tokens[start:start + size])
                matched, score = match_customer(candidate, "", directory)
                if matched and score > best_score:
                    best = matched
                    best_score = score
        if best:
            return str(best.get("name", "") or "").strip()

    return clean_company_display_name(prefix) or prefix


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


def _company_phonetic_fold(value: str):
    value = normalize_company(value)
    replacements = (
        ("sch", "sz"), ("ch", "h"), ("rz", "z"), ("ż", "z"), ("ź", "z"),
        ("ó", "u"), ("ą", "on"), ("ę", "en"), ("v", "w"), ("q", "k"),
        ("x", "ks"), ("ph", "f"), ("th", "t"),
    )
    for old, new in replacements:
        value = value.replace(old, new)
    return re.sub(r"[^a-z0-9]", "", value)


def rank_company_candidates(company: str, directory: list):
    source = normalize_company(company)
    if not source:
        return []

    source_compact = source.replace(" ", "")
    source_phonetic = _company_phonetic_fold(source)
    source_tokens = source.split()
    ranked = []

    for item in directory:
        variants = [item.get("name", ""), *(item.get("aliases") or [])]
        best = 0.0
        best_variant = ""
        for variant in variants:
            target = normalize_company(variant)
            if not target:
                continue
            target_compact = target.replace(" ", "")
            target_phonetic = _company_phonetic_fold(target)

            score = SequenceMatcher(None, source, target).ratio()
            if source_compact and target_compact:
                score = max(score, SequenceMatcher(None, source_compact, target_compact).ratio())
            if source_phonetic and target_phonetic:
                score = max(score, SequenceMatcher(None, source_phonetic, target_phonetic).ratio())

            target_tokens = target.split()
            if source_tokens and target_tokens:
                token_scores = [
                    SequenceMatcher(None, s, t).ratio()
                    for s in source_tokens
                    for t in target_tokens
                    if s and t
                ]
                if token_scores:
                    score = max(score, max(token_scores) * 0.94)

            if source == target:
                score = 1.0
            elif source in target or target in source:
                score = max(score, 0.94)

            if score > best:
                best = score
                best_variant = variant

        if best > 0:
            ranked.append((best, item, best_variant))

    ranked.sort(key=lambda row: row[0], reverse=True)
    return ranked


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

    ranked = rank_company_candidates(company, directory)
    if not ranked:
        return None, 0.0

    best_score, best, _ = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    margin = best_score - second_score
    compact_len = len(source.replace(" ", ""))
    dictionary_entry = bool(best.get("from_company_dictionary"))

    if dictionary_entry:
        if compact_len <= 6:
            accepted = best_score >= 0.54 and margin >= 0.14
        else:
            accepted = best_score >= 0.60 and margin >= 0.10
    elif compact_len <= 6:
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
        base_customers = parse_customer_directory(self.settings.get("customer_directory", ""))
        alias_items = parse_company_alias_dictionary(self.settings.get("company_alias_dictionary", ""))
        self.customer_directory = merge_company_alias_dictionary(base_customers, alias_items)
        self.problem_dictionary = parse_problem_dictionary(self.settings.get("problem_dictionary", ""))
        self.confirmation_pending = False
        self.awaiting_correction = False
        self.correction_field = ""
        self.correction_attempts = 0
        self.correction_choice_misses = 0
        self.correction_rejected_pending = False
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
        self.rejected_company_counts = {}
        self.company_rejection_total = 0
        self.company_recognition_failures = 0
        self.early_problem_score = None
        self.problem_recognition_failures = 0
        self.company_trusted = False
        self.contact_trusted = False
        self.contact_attempts = 0
        self.awaiting_contact_dtmf = False
        self.dtmf_contact_buffer = ""
        self.dtmf_contact_context = ""
        self.awaiting_confirmation_dtmf = ""
        self.awaiting_company_phone_recovery = False
        self.cancellation_suspected = False
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

    def ticket_has_untrusted_fields(self):
        for field in ("company", "contact", "description"):
            meta = self.ticket_field_meta.get(field) or {}
            if meta.get("trusted") is False:
                return True
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

        # Last-resort guard: never send a known prompt/noise transcript as a
        # company name, even if an unusual path bypassed earlier validation.
        raw_company = str(ticket.get("company", "") or "").strip()
        if raw_company and looks_like_invalid_company_name(raw_company):
            ticket["company"] = "Nazwa firmy nierozpoznana – zweryfikować"
            uncertain = True
            self.ticket_field_meta["company"] = {
                "source": "invalid_stt_guard",
                "trusted": False,
            }
            warning_text = (
                warning_text.strip() + "\n"
                if warning_text.strip()
                else ""
            ) + "UWAGA: Nazwa firmy nie została wiarygodnie rozpoznana i wymaga weryfikacji."

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
        if self.awaiting_company_phone_recovery:
            return "contact"
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
        # Do not seed field-specific Whisper decodes with full sentences.
        # initial_prompt becomes previous-text context and was the primary source
        # of verbatim prompt leaks in short 8 kHz telephone utterances.
        if (
            self.confirmation_pending
            or self.company_confirmation_pending
            or self.awaiting_company
            or self.awaiting_contact
            or self.awaiting_company_phone_recovery
            or self.awaiting_problem
            or self.awaiting_correction
        ):
            return ""
        return (self.settings.get("stt_prompt", "") or "").strip()

    def stt_hotwords_for_state(self):
        customer_names = " ".join(
            str(x.get("name", "") or "").strip()
            for x in self.customer_directory
            if str(x.get("name", "") or "").strip()
        )

        if self.confirmation_pending or self.company_confirmation_pending:
            return "tak nie"

        if self.awaiting_correction and not self.correction_field:
            return "firma nazwa numer telefon kontakt opis problem"

        if self.awaiting_correction and self.correction_field == "company":
            return ""

        if self.awaiting_company_phone_recovery:
            return "0 1 2 3 4 5 6 7 8 9"

        if self.awaiting_company:
            return ""

        if self.awaiting_correction and self.correction_field == "contact":
            return "0 1 2 3 4 5 6 7 8 9"

        if self.awaiting_contact:
            return "0 1 2 3 4 5 6 7 8 9"

        if self.awaiting_correction and self.correction_field == "description":
            if self.problem_dictionary:
                return " ".join(self.problem_dictionary)
            return (self.settings.get("stt_problem_hint", "") or "").strip()

        if self.awaiting_problem:
            if self.problem_dictionary:
                return " ".join(self.problem_dictionary)
            return (self.settings.get("stt_problem_hint", "") or "").strip()

        return ""

    async def handle_dtmf(self, payload: bytes):
        try:
            chars = payload.decode("ascii", errors="ignore")
        except Exception:
            chars = ""

        if self.awaiting_confirmation_dtmf:
            for ch in chars:
                mode = self.awaiting_confirmation_dtmf
                allowed = ("1", "2", "3") if mode == "correction_choice" else ("1", "2")
                if ch not in allowed:
                    continue
                self.awaiting_confirmation_dtmf = ""
                self.confirmation_misses = 0
                if mode == "company":
                    if ch == "1":
                        self.ticket_data["company"] = self.company_candidate
                        self.company_trusted = True
                        self.mark_ticket_field("company", "confirmed_dtmf", self.company_candidate_score, True)
                        self.company_confirmation_pending = False
                        self.company_candidate = ""
                        self.company_candidate_score = None
                        self.company_confirmation_context = ""
                        self.awaiting_company = False
                        if self.ticket_data.get("contact"):
                            self.awaiting_problem = True
                            await self.say("Dziękuję. Proszę opisać problem.")
                        else:
                            self.awaiting_contact = True
                            await self.say("Dziękuję. Proszę podać numer telefonu kontaktowego.")
                    else:
                        self.company_rejection_total += 1
                        self.company_confirmation_pending = False
                        self.company_candidate = ""
                        self.company_candidate_score = None
                        self.company_confirmation_context = ""
                        self.awaiting_company = True
                        if self.company_rejection_total >= 2:
                            self.awaiting_company_phone_recovery = True
                            await self.say("Proszę podać numer telefonu kontaktowego, spróbuję odnaleźć firmę.")
                        else:
                            await self.say("Dobrze. Proszę podać samą nazwę firmy jeszcze raz.")
                    return
                if mode == "correction_choice":
                    self.correction_choice_misses = 0
                    self.awaiting_correction = True
                    self.correction_field = {"1": "company", "2": "contact", "3": "description"}[ch]
                    if self.correction_field == "company":
                        await self.say("Proszę podać poprawną nazwę firmy.")
                    elif self.correction_field == "contact":
                        await self.say("Proszę podać poprawny numer telefonu kontaktowego.")
                    else:
                        await self.say("Proszę podać poprawny opis problemu.")
                    return
                if mode == "ticket":
                    if ch == "1":
                        self.confirmation_pending = False
                        await self.finalize_ticket()
                    else:
                        self.confirmation_pending = False
                        self.correction_rejected_pending = True
                        self.correction_choice_misses = 0
                        self.awaiting_correction = True
                        self.correction_field = "description" if self.caller_matched_customer else ""
                        if self.correction_field == "description":
                            await self.say("Dobrze. Proszę podać poprawny opis problemu.")
                        else:
                            await self.say(
                                "Dobrze. Proszę podać ponownie tylko dane, które mam poprawić: "
                                "nazwę firmy, numer kontaktowy albo opis problemu."
                            )
                    return
            return

        if not self.awaiting_contact_dtmf:
            return
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
                    if matched_customer.get("from_company_dictionary") and match_score < 0.86:
                        self.company_confirmation_pending = True
                        self.company_candidate = str(matched_customer.get("name", "") or "").strip()
                        self.company_candidate_score = float(selected_score) if selected_score is not None else None
                        self.company_confirmation_context = "correction"
                        self.company_candidate_phone = matched_customer.get("phone") or ""
                        await self.say(
                            f"Czy chodzi o firmę {self.company_candidate}? "
                            "Proszę powiedzieć tak albo nie."
                        )
                        return
                    self.ticket_data["company"] = matched_customer["name"]
                    self.company_trusted = True
                    self.mark_ticket_field("company", "directory", match_score, True)
                    if matched_customer.get("phone"):
                        self.ticket_data["contact"] = matched_customer["phone"]
                        self.contact_trusted = True
                        self.mark_ticket_field("contact", "directory", None, True)
                if context == "company_recovery":
                    if matched_customer:
                        self.awaiting_company_phone_recovery = False
                        self.company_recognition_failures = 0
                        self.awaiting_company = False
                        if self.ticket_data.get("description"):
                            self.confirmation_pending = True
                            await self.say_confirmation_summary()
                        else:
                            self.awaiting_problem = True
                            await self.say("Dziękuję. Proszę opisać problem.")
                    else:
                        self.awaiting_company_phone_recovery = False
                        self.awaiting_company = True
                        self.company_rejection_total = 0
                        await self.say("Nie znalazłem firmy po tym numerze. Proszę podać nazwę firmy.")
                    return
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
                self.stt_hotwords_for_state(),
            )
            if isinstance(stt_result, dict):
                text = str(stt_result.get("selected", "") or "")
                selected_score = stt_result.get("selected_score")
                if stt_result.get("mode") == "contact" and text:
                    raw_contact_text = text
                    normalized_contact = extract_phone_digits(
                        text,
                        self.settings.get("phone_validation_mode", "pl"),
                    )
                    if normalized_contact:
                        text = normalized_contact
                        stt_result["selected_raw"] = raw_contact_text
                        stt_result["selected"] = normalized_contact
                        stt_result["normalized_contact"] = True
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

            if retry_reason == "residual_prompt_artifact":
                log.info("[%s] Ignoring residual prompt artifact after TTS", self.call_id)
                return

            # Real company speech that produced no usable transcript still counts
            # toward escaping the company-name loop. Ignore the ~0.6 s residual
            # prompt echo right after TTS.
            if (
                self.awaiting_company
                and not self.awaiting_company_phone_recovery
                and audio_seconds >= 0.90
                and retry_reason != "short_rejected_audio"
            ):
                self.company_recognition_failures += 1
                if self.company_recognition_failures >= 3:
                    self.awaiting_company_phone_recovery = True
                    self.contact_attempts = 0
                    self.stt_misses = 0
                    await self.say(
                        "Nie udało mi się pewnie rozpoznać nazwy firmy. "
                        "Proszę podać numer telefonu kontaktowego, spróbuję odnaleźć firmę."
                    )
                    return

            if self.awaiting_correction and not self.correction_field and audio_seconds >= 0.50:
                self.correction_choice_misses += 1
                if self.correction_choice_misses >= 2:
                    self.correction_choice_misses = 0
                    self.awaiting_confirmation_dtmf = "correction_choice"
                    await self.say(
                        "Proszę nacisnąć 1 dla nazwy firmy, 2 dla numeru kontaktowego "
                        "albo 3 dla opisu problemu."
                    )
                    return

            # Rejected phone speech is still a failed contact attempt. Do not
            # allow a different hallucination class to create an infinite loop.
            if (self.awaiting_contact or self.awaiting_company_phone_recovery) and audio_seconds >= 0.50 and retry_reason != "short_rejected_audio":
                self.contact_attempts += 1
                if self.contact_attempts >= 2:
                    self.awaiting_contact_dtmf = True
                    self.dtmf_contact_buffer = ""
                    self.dtmf_contact_context = "company_recovery" if self.awaiting_company_phone_recovery else "initial"
                    self.stt_misses = 0
                    await self.say(
                        "Nie udało mi się pewnie rozpoznać numeru. "
                        "Proszę wpisać dziewięć cyfr na klawiaturze telefonu i zakończyć krzyżykiem."
                    )
                    return
            if (self.company_confirmation_pending or self.confirmation_pending) and audio_seconds >= 0.50:
                self.confirmation_misses += 1
                if self.confirmation_misses >= 2:
                    self.confirmation_misses = 0
                    self.awaiting_confirmation_dtmf = "company" if self.company_confirmation_pending else "ticket"
                    await self.say("Proszę nacisnąć 1, jeśli tak, albo 2, jeśli nie.")
                    return
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

        # Remember a likely cancellation even if STT mangled one keyword. This
        # prevents the hangup fallback from creating an unwanted ticket.
        if looks_like_possible_cancellation(text):
            self.cancellation_suspected = True

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
            "do zobaczenia",
            "dziękuję do zobaczenia",
            "dziekuje do zobaczenia",
            "to wszystko",
            "koniec",
        )
        if any(phrase in normalized for phrase in goodbye_phrases):
            in_problem_state = self.awaiting_problem or (
                self.awaiting_correction and self.correction_field == "description"
            )
            try:
                weak_goodbye = selected_score is not None and float(selected_score) < -0.45
            except (TypeError, ValueError):
                weak_goodbye = False
            suspicious_problem_goodbye = (
                in_problem_state
                and (
                    weak_goodbye
                    or "wszystko w porządku" in normalized
                    or "wszystko w porzadku" in normalized
                )
            )
            if suspicious_problem_goodbye:
                self.problem_recognition_failures += 1
                await self.say(
                    "Nie udało mi się wiarygodnie rozpoznać opisu problemu. Proszę powtórzyć."
                )
                return

            # Clear local confirmation states before ending the call so the
            # session cannot remain logically pending after a goodbye.
            self.company_confirmation_pending = False
            self.confirmation_pending = False
            if self.has_complete_ticket_data() and not self.ticket_ref and not self.cancellation_suspected:
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
                try:
                    weak_yes = selected_score is not None and float(selected_score) < -0.45
                except (TypeError, ValueError):
                    weak_yes = False
                try:
                    weak_candidate = (
                        self.company_candidate_score is not None
                        and float(self.company_candidate_score) < float(self.settings.get("company_confirm_logprob", -0.55))
                    )
                except (TypeError, ValueError):
                    weak_candidate = False
                if weak_yes and weak_candidate:
                    self.awaiting_confirmation_dtmf = "company"
                    await self.say("Dla pewności proszę nacisnąć 1, jeśli tak, albo 2, jeśli nie.")
                    return
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
                    self.rejected_company_counts[rejected_normalized] = (
                        self.rejected_company_counts.get(rejected_normalized, 0) + 1
                    )
                    self.company_rejection_total += 1
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
                rejected_count = self.rejected_company_counts.get(rejected_normalized, 0)
                should_recover_by_phone = rejected_count >= 2 or self.company_rejection_total >= 2
                if context == "correction":
                    self.awaiting_correction = True
                    self.correction_field = "company"
                    if should_recover_by_phone:
                        await self.say(
                            "Ta nazwa została już dwa razy odrzucona. "
                            "Proszę podać inną nazwę firmy."
                        )
                    else:
                        await self.say("Dobrze. Proszę podać poprawną nazwę firmy jeszcze raz.")
                else:
                    self.awaiting_company = True
                    if should_recover_by_phone:
                        self.awaiting_company_phone_recovery = True
                        await self.say(
                            "Ta nazwa została już dwa razy odrzucona. "
                            "Proszę podać numer telefonu kontaktowego, spróbuję odnaleźć firmę."
                        )
                    else:
                        await self.say("Dobrze. Proszę podać samą nazwę firmy jeszcze raz.")
                return

            self.confirmation_misses += 1
            if self.confirmation_misses >= 2:
                self.confirmation_misses = 0
                self.company_rejection_total += 1
                if self.company_rejection_total >= 2:
                    self.company_confirmation_pending = False
                    self.company_candidate = ""
                    self.company_candidate_score = None
                    self.company_confirmation_context = ""
                    self.awaiting_company = True
                    self.awaiting_company_phone_recovery = True
                    await self.say("Proszę podać numer telefonu kontaktowego, spróbuję odnaleźć firmę.")
                    return
                self.awaiting_confirmation_dtmf = "company"
                await self.say("Proszę nacisnąć 1, jeśli tak, albo 2, jeśli nie.")
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
                try:
                    weak_yes = selected_score is not None and float(selected_score) < -0.45
                except (TypeError, ValueError):
                    weak_yes = False
                if weak_yes and self.ticket_has_untrusted_fields():
                    self.awaiting_confirmation_dtmf = "ticket"
                    await self.say("Dla pewności proszę nacisnąć 1, jeśli dane są poprawne, albo 2, jeśli nie.")
                    return
                self.confirmation_pending = False
                self.confirmation_misses = 0
                await self.finalize_ticket()
                return

            if matches_confirmation_phrase(normalized, no_phrases):
                self.confirmation_pending = False
                self.confirmation_misses = 0
                self.correction_rejected_pending = True
                self.correction_choice_misses = 0
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

            # Ambiguous confirmation must never be guessed by the LLM. A
            # misheard "tak" must not turn into a correction request.
            self.confirmation_misses += 1
            if self.confirmation_misses >= 2:
                self.confirmation_misses = 0
                self.awaiting_confirmation_dtmf = "ticket"
                await self.say("Proszę nacisnąć 1, jeśli tak, albo 2, jeśli nie.")
                return
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
                words = set(re.findall(r"[a-ząćęłńóśźż]+", normalized, flags=re.IGNORECASE))
                company_choice = any(word.startswith(("firm", "nazw")) for word in words)
                contact_choice = any(word.startswith(("numer", "telefon", "kontakt")) for word in words)
                description_choice = any(word.startswith(("opis", "problem")) for word in words)
                choices = sum((company_choice, contact_choice, description_choice))

                if choices == 1 and company_choice:
                    self.correction_choice_misses = 0
                    self.correction_field = "company"
                    await self.say("Proszę podać poprawną nazwę firmy.")
                    return
                if choices == 1 and contact_choice:
                    self.correction_choice_misses = 0
                    self.correction_field = "contact"
                    await self.say("Proszę podać poprawny numer telefonu kontaktowego.")
                    return
                if choices == 1 and description_choice:
                    self.correction_choice_misses = 0
                    self.correction_field = "description"
                    await self.say("Proszę podać poprawny opis problemu.")
                    return

                self.correction_choice_misses += 1
                if self.correction_choice_misses >= 2:
                    self.correction_choice_misses = 0
                    self.awaiting_confirmation_dtmf = "correction_choice"
                    await self.say(
                        "Proszę nacisnąć 1 dla nazwy firmy, 2 dla numeru kontaktowego "
                        "albo 3 dla opisu problemu."
                    )
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
                    ranked_candidates = rank_company_candidates(company_text, self.customer_directory)
                    if ranked_candidates:
                        best_score, best_item, _ = ranked_candidates[0]
                        second_score = ranked_candidates[1][0] if len(ranked_candidates) > 1 else 0.0
                        margin = best_score - second_score
                        if (
                            best_item.get("from_company_dictionary")
                            and best_score >= 0.42
                            and margin >= 0.10
                        ):
                            self.company_confirmation_pending = True
                            self.company_candidate = str(best_item.get("name", "") or "").strip()
                            self.company_candidate_score = float(selected_score) if selected_score is not None else None
                            self.company_confirmation_context = "correction"
                            self.company_candidate_phone = best_item.get("phone") or ""
                            await self.say(
                                f"Czy chodzi o firmę {self.company_candidate}? "
                                "Proszę powiedzieć tak albo nie."
                            )
                            return

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
                invalid_problem = looks_like_invalid_problem_description(text)
                try:
                    very_low_problem = selected_score is not None and float(selected_score) < -0.80
                except (TypeError, ValueError):
                    very_low_problem = False
                if invalid_problem or very_low_problem:
                    self.problem_recognition_failures += 1
                    if self.problem_recognition_failures < 2:
                        await self.say("Nie udało mi się wiarygodnie rozpoznać opisu problemu. Proszę powtórzyć.")
                        return
                    replacement = (
                        "Opis problemu nierozpoznany – zweryfikować"
                        if invalid_problem else text.strip()
                    )
                    self.ticket_data["description"] = replacement
                    self.mark_ticket_field("description", "stt_uncertain", selected_score, False)
                    self.ticket_data["title"] = replacement[:80]
                    self.awaiting_correction = False
                    self.correction_field = ""
                    self.correction_rejected_pending = False
                    await self.finalize_ticket(
                        uncertain=True,
                        warning_text="UWAGA: Poprawiony opis problemu pozostał niepewny po dwóch próbach."
                    )
                    return
                self.problem_recognition_failures = 0
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
            self.correction_rejected_pending = False
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
        if self.awaiting_company:
            phone = extract_phone_digits(text, self.settings.get("phone_validation_mode", "pl"))
            company_text = clean_company_display_name(text) or text.strip()

            if self.awaiting_company_phone_recovery:
                numericish = bool(re.search(r"\d", text)) or bool(phone)
                if phone:
                    phone_customer, phone_score = match_customer("", phone, self.customer_directory)
                    if phone_customer:
                        self.ticket_data["company"] = phone_customer["name"]
                        self.ticket_data["contact"] = phone_customer.get("phone") or phone
                        self.company_trusted = True
                        self.contact_trusted = True
                        self.mark_ticket_field("company", "directory", phone_score, True)
                        self.mark_ticket_field("contact", "directory", None, True)
                        self.record_customer_match("company_recovery_phone", "", phone, phone_customer, phone_score)
                        self.awaiting_company_phone_recovery = False
                        self.awaiting_company = False
                        self.contact_attempts = 0
                        self.awaiting_problem = True
                        await self.say("Dziękuję. Proszę opisać problem.")
                        return

                # We explicitly asked for a phone number. Never reinterpret a
                # non-number response as another company name; that created a
                # recovery loop when callers said e.g. "momencik, nie pamiętam".
                self.contact_attempts += 1
                if self.contact_attempts >= 2:
                    self.awaiting_contact_dtmf = True
                    self.dtmf_contact_buffer = ""
                    self.dtmf_contact_context = "company_recovery"
                    await self.say(
                        "Nie udało mi się pewnie rozpoznać numeru. "
                        "Proszę wpisać dziewięć cyfr na klawiaturze telefonu i zakończyć krzyżykiem."
                    )
                elif numericish:
                    await self.say("Nie znalazłem firmy po tym numerze. Proszę podać numer jeszcze raz.")
                else:
                    self.awaiting_contact_dtmf = True
                    self.dtmf_contact_buffer = ""
                    self.dtmf_contact_context = "company_recovery"
                    await self.say(
                        "Nie udało mi się rozpoznać numeru telefonu. "
                        "Proszę podać numer albo wpisać go na klawiaturze telefonu."
                    )
                return

            if phone:
                phone_customer, phone_score = match_customer("", phone, self.customer_directory)
                if phone_customer:
                    self.ticket_data["company"] = phone_customer["name"]
                    self.ticket_data["contact"] = phone_customer.get("phone") or phone
                    self.company_trusted = True
                    self.contact_trusted = True
                    self.mark_ticket_field("company", "directory", phone_score, True)
                    self.mark_ticket_field("contact", "directory", None, True)
                    self.record_customer_match("company_recovery_phone", "", phone, phone_customer, phone_score)
                    self.company_recognition_failures = 0
                    self.awaiting_company = False
                    self.awaiting_problem = True
                    await self.say("Dziękuję. Proszę opisać problem.")
                    return

            if looks_like_invalid_company_name(company_text):
                self.company_recognition_failures += 1
                if self.company_recognition_failures >= 3:
                    self.awaiting_company_phone_recovery = True
                    self.contact_attempts = 0
                    await self.say(
                        "Nie udało mi się pewnie rozpoznać nazwy firmy. "
                        "Proszę podać numer telefonu kontaktowego, spróbuję odnaleźć firmę."
                    )
                    return
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

                # Keep this hot path deterministic. Calling the local LLM here
                # previously added up to ~41 s despite company/problem already
                # being extractable from the utterance.
                company_text = extract_company_fragment(text, self.customer_directory)

            self.company_recognition_failures = 0
            matched_customer, match_score = match_customer(
                company_text,
                phone,
                self.customer_directory,
            )
            self.record_customer_match("initial_company", company_text, phone, matched_customer, match_score)
            if matched_customer:
                if matched_customer.get("from_company_dictionary") and match_score < 0.86:
                    self.company_confirmation_pending = True
                    self.company_candidate = str(matched_customer.get("name", "") or "").strip()
                    self.company_candidate_score = float(selected_score) if selected_score is not None else None
                    self.company_confirmation_context = "initial"
                    self.company_candidate_phone = matched_customer.get("phone") or ""
                    self.awaiting_company = False
                    await self.say(
                        f"Czy chodzi o firmę {self.company_candidate}? "
                        "Proszę powiedzieć tak albo nie."
                    )
                    return
                self.ticket_data["company"] = matched_customer["name"]
                self.company_trusted = True
                self.mark_ticket_field("company", "directory", match_score, True)
                if matched_customer.get("phone"):
                    self.ticket_data["contact"] = matched_customer["phone"]
                    self.contact_trusted = True
                    self.mark_ticket_field("contact", "directory", None, True)
            else:
                ranked_candidates = rank_company_candidates(company_text, self.customer_directory)
                if ranked_candidates:
                    best_score, best_item, _ = ranked_candidates[0]
                    second_score = ranked_candidates[1][0] if len(ranked_candidates) > 1 else 0.0
                    margin = best_score - second_score
                    if (
                        best_item.get("from_company_dictionary")
                        and best_score >= 0.42
                        and margin >= 0.10
                    ):
                        self.company_confirmation_pending = True
                        self.company_candidate = str(best_item.get("name", "") or "").strip()
                        self.company_candidate_score = float(selected_score) if selected_score is not None else None
                        self.company_confirmation_context = "initial"
                        self.company_candidate_phone = best_item.get("phone") or ""
                        self.awaiting_company = False
                        await self.say(
                            f"Czy chodzi o firmę {self.company_candidate}? "
                            "Proszę powiedzieć tak albo nie."
                        )
                        return

                confirm_threshold = float(self.settings.get("company_confirm_logprob", -0.55))
                low_confidence = (
                    selected_score is not None
                    and float(selected_score) < confirm_threshold
                )
                normalized_candidate = normalize_company(company_text)
                was_rejected = normalized_candidate in self.rejected_company_names
                rejected_count = self.rejected_company_counts.get(normalized_candidate, 0)
                if self.company_rejection_total >= 2 or (was_rejected and rejected_count >= 2):
                    self.awaiting_company = True
                    self.awaiting_company_phone_recovery = True
                    await self.say(
                        "Ta nazwa była już odrzucona. "
                        "Proszę podać numer telefonu kontaktowego, spróbuję odnaleźć firmę."
                    )
                    return
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

            invalid_problem = looks_like_invalid_problem_description(text)
            try:
                very_low_problem = selected_score is not None and float(selected_score) < -0.80
            except (TypeError, ValueError):
                very_low_problem = False
            if invalid_problem or very_low_problem:
                self.problem_recognition_failures += 1
                if self.problem_recognition_failures < 2:
                    await self.say("Nie udało mi się wiarygodnie rozpoznać opisu problemu. Proszę powtórzyć.")
                    return
                description = (
                    "Opis problemu nierozpoznany – zweryfikować"
                    if invalid_problem else text.strip()
                )
                self.ticket_data["description"] = description
                self.early_problem_score = selected_score
                self.mark_ticket_field("description", "stt_uncertain", selected_score, False)
                self.awaiting_problem = False
                self.ticket_data["title"] = description[:80]
                if self.has_complete_ticket_data():
                    await self.finalize_ticket(
                        uncertain=True,
                        warning_text="UWAGA: Opis problemu pozostał niepewny po dwóch próbach."
                    )
                    return
            self.problem_recognition_failures = 0
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

        # Deterministic recovery path. Normal ticket collection should already
        # have returned from an explicit state above; this branch exists only as
        # a safety net for stale/legacy state combinations. It deliberately does
        # not call an LLM.
        company_ok = bool(str(self.ticket_data.get("company", "") or "").strip())
        contact_ok = bool(str(self.ticket_data.get("contact", "") or "").strip())
        description_ok = bool(str(self.ticket_data.get("description", "") or "").strip())

        if self.turns >= int(self.settings.get("max_turns", 8)):
            await self.say(
                "Nie udało się zebrać kompletu informacji. Proszę skontaktować się z serwisem."
            )
            self.final_status = "incomplete"
            self.closed = True
            self.writer.close()
            return

        self.awaiting_company = False
        self.awaiting_contact = False
        self.awaiting_problem = False

        if not company_ok:
            self.awaiting_company = True
            await self.say("Proszę podać nazwę firmy.")
            return

        if not contact_ok:
            self.awaiting_contact = True
            await self.say("Proszę podać numer telefonu kontaktowego.")
            return

        if not description_ok:
            self.awaiting_problem = True
            await self.say("Proszę opisać problem.")
            return

        description = str(self.ticket_data.get("description", "") or "").strip()
        if not self.ticket_data.get("title"):
            self.ticket_data["title"] = description[:80] or "Zgłoszenie telefoniczne"

        if self.can_auto_finalize(self.early_problem_score):
            self.confirmation_pending = False
            self.confirmation_misses = 0
            await self.finalize_ticket()
            return

        self.confirmation_pending = True
        self.confirmation_misses = 0
        await self.say_confirmation_summary()
        return

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
            and session.final_status not in ("completed", "completed_uncertain", "icp_error", "caller_cancelled")
            and not session.cancellation_suspected
            and not session.correction_rejected_pending
            and not session.awaiting_correction
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
