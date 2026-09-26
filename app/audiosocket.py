import asyncio
import json
import logging
import re
import struct
import time
import uuid
from collections import deque
from difflib import SequenceMatcher
import webrtcvad

from .config import load_settings, decrypt_secret
from .stt import transcribe_pcm16
from .tts import synthesize_pcm8k
from .llm import ask_ollama, interpret_turn
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
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def speak_phone(value: str):
    digits = re.sub(r"\D", "", value or "")
    return " ".join(digits) if digits else value


def extract_phone_digits(value: str):
    digits = re.sub(r"\D", "", value or "")
    return digits if 9 <= len(digits) <= 15 else ""


def matches_confirmation_phrase(text: str, phrases: tuple[str, ...]):
    normalized = " ".join((text or "").lower().strip(" .,!?:;").split())
    if not normalized:
        return False
    return normalized in phrases


def company_without_phone(value: str):
    cleaned = re.sub(r"[\d\s,.;:+()\-]{7,}", " ", value or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:-")
    return cleaned.strip()


def match_customer(company: str, contact: str, directory: list):
    contact_digits = re.sub(r"\D", "", contact or "")
    # Phone match is strongest and can repair a badly recognized company name.
    if contact_digits:
        for item in directory:
            phone = item.get("phone", "")
            if phone and (contact_digits == phone or contact_digits.endswith(phone[-9:]) or phone.endswith(contact_digits[-9:])):
                return item, 1.0

    source = normalize_company(company)
    if not source:
        return None, 0.0

    best = None
    best_score = 0.0
    for item in directory:
        target = normalize_company(item.get("name", ""))
        if not target:
            continue
        score = SequenceMatcher(None, source, target).ratio()
        if source in target or target in source:
            score = max(score, 0.88)
        if score > best_score:
            best, best_score = item, score
    return (best, best_score) if best_score >= 0.58 else (None, best_score)


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
        self.customer_directory = parse_customer_directory(self.settings.get("customer_directory", ""))
        self.confirmation_pending = False
        self.awaiting_correction = False
        self.awaiting_company = False
        self.awaiting_contact = False
        self.awaiting_problem = False
        self.stt_misses = 0
        self.listen_not_before = 0.0
        self.last_tts_end = 0.0
        self.last_repeat_prompt = 0.0
        self.confirmation_misses = 0
        self.last_question = ""
        self.final_status = "ended"
        self.ticket_ref = ""
        self.final_error = ""

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
        self.listen_not_before = self.last_tts_end + (0.20 if self.confirmation_pending else 0.45)

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
        await self.say("Czy dane są poprawne? Proszę powiedzieć tak lub nie.")

    async def finalize_ticket(self):
        ticket = dict(self.ticket_data)
        try:
            token = decrypt_secret(self.settings.get("icp_token_enc", ""))
            client = ICProjectClient(
                self.settings.get("icp_instance", ""),
                token,
                self.settings.get("icp_board_column", ""),
            )
            created = await client.create_task(ticket, self.settings.get("icp_priority", "normal"))
            ticket_no = created.get("number") or created.get("shortCode") or ""
            suffix = f" Numer zgłoszenia: {ticket_no}." if ticket_no else ""
            self.ticket_ref = str(ticket_no or created.get("id") or "")
            self.final_status = "completed"
            await self.say("Dziękuję. Zgłoszenie zostało zapisane." + suffix + " Do widzenia.")
            self.closed = True
            await asyncio.sleep(0.3)
            self.writer.close()
            return True
        except Exception as e:
            self.final_status = "icp_error"
            self.final_error = str(e)
            log.exception("[%s] ICP create error", self.call_id)
            await self.say(
                "Nie udało się zapisać zgłoszenia w systemie. "
                "Proszę skontaktować się z serwisem. Do widzenia."
            )
            self.closed = True
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
                silence_target_ms = 350 if self.confirmation_pending else int(self.settings.get("silence_ms", 900))
                if silence_ms >= silence_target_ms:
                    pcm = bytes(self.speech)
                    self.speech.clear()
                    self.speaking = False
                    self.silence_frames = 0
                    if len(pcm) >= 320 * 15:
                        await self.process_utterance(pcm)

    async def process_utterance(self, pcm: bytes):
        self.turns += 1
        try:
            text = await asyncio.to_thread(
                transcribe_pcm16,
                pcm,
                self.settings["whisper_model"],
                self.settings["whisper_device"],
                self.settings["whisper_compute_type"],
                8000,
                (
                    self.settings.get("stt_prompt", "")
                    + (" Klienci: " + ", ".join(x["name"] for x in self.customer_directory) if self.customer_directory else "")
                ),
            )
        except Exception:
            log.exception("[%s] STT error", self.call_id)
            await self.say("Nie udało mi się rozpoznać wypowiedzi. Proszę powtórzyć.")
            return

        if not text:
            self.stt_misses += 1
            now = time.monotonic()
            wait_before_repeat = 2.0 if self.confirmation_pending else 4.0
            enough_time_to_answer = (now - self.last_tts_end) >= wait_before_repeat
            repeat_cooldown_ok = (now - self.last_repeat_prompt) >= 8.0
            if self.stt_misses >= 3 and enough_time_to_answer and repeat_cooldown_ok:
                self.stt_misses = 0
                self.last_repeat_prompt = now
                await self.say("Nie dosłyszałem. Proszę powtórzyć.")
            return

        self.stt_misses = 0

        log.info("[%s] STT: %s", self.call_id, text)
        add_message(self.call_id, "user", text)
        self.history.append({"role": "user", "content": text})

        normalized = " ".join(text.lower().strip(" .,!?:;").split())
        goodbye_phrases = (
            "do widzenia",
            "dziękuję do widzenia",
            "dziekuje do widzenia",
            "to wszystko",
            "koniec",
        )
        if any(phrase in normalized for phrase in goodbye_phrases):
            self.final_status = "caller_ended"
            await self.say("Dziękuję za rozmowę. Do widzenia.")
            self.closed = True
            await asyncio.sleep(0.2)
            self.writer.close()
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
                await self.say(
                    "Dobrze. Proszę podać ponownie tylko dane, które mam poprawić: "
                    "nazwę firmy, numer kontaktowy albo opis problemu."
                )
                return

            # Ambiguous confirmation must never create a ticket.
            # LLM may help detect a correction/negative intent, but it is not
            # allowed to turn an unclear transcript into a positive confirmation.
            interpreted = await self.interpret_fallback("potwierdzenie danych tak/nie", text)
            intent = str(interpreted.get("intent", "")).lower()

            if intent in ("confirm_no", "correction"):
                self.confirmation_pending = False
                self.confirmation_misses = 0
                self.awaiting_correction = True
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
                    "Proszę powiedzieć tylko: tak albo nie."
                )
            return

        # Fast deterministic state machine for normal calls.
        # Ollama is only a fallback for corrections / unusual utterances.
        if self.awaiting_company:
            phone = extract_phone_digits(text)
            company_text = company_without_phone(text) or text.strip()

            looks_like_problem = any(
                phrase in normalized
                for phrase in ("problem", "nie działa", "nie dziala", "awaria", "błąd", "blad", "usterka", "nie mogę", "nie moge")
            )
            if looks_like_problem:
                interpreted = await self.interpret_fallback("nazwa firmy", text)
                if interpreted.get("company"):
                    company_text = str(interpreted["company"]).strip()
                if interpreted.get("contact") and not phone:
                    phone = extract_phone_digits(str(interpreted["contact"]))
                if interpreted.get("description"):
                    self.ticket_data["description"] = str(interpreted["description"]).strip()

            matched_customer, match_score = match_customer(
                company_text,
                phone,
                self.customer_directory,
            )
            if matched_customer:
                self.ticket_data["company"] = matched_customer["name"]
                if matched_customer.get("phone"):
                    self.ticket_data["contact"] = matched_customer["phone"]
            else:
                self.ticket_data["company"] = company_text

            if phone and not self.ticket_data.get("contact"):
                self.ticket_data["contact"] = phone

            self.awaiting_company = False

            if self.ticket_data.get("contact"):
                self.awaiting_problem = True
                await self.say("Dziękuję. Proszę opisać problem.")
            else:
                self.awaiting_contact = True
                await self.say("Dziękuję. Proszę podać numer telefonu kontaktowego.")
            return

        if self.awaiting_contact:
            phone = extract_phone_digits(text)
            if not phone:
                interpreted = await self.interpret_fallback("numer telefonu kontaktowego", text)
                phone = extract_phone_digits(str(interpreted.get("contact", "") or ""))

                if not phone and interpreted.get("intent") == "problem" and interpreted.get("description"):
                    self.ticket_data["description"] = str(interpreted["description"]).strip()

                if not phone and interpreted.get("company"):
                    self.ticket_data["company"] = str(interpreted["company"]).strip()

                if not phone:
                    await self.say(
                        "Nie udało mi się rozpoznać numeru telefonu. "
                        "Proszę podać go cyfra po cyfrze."
                    )
                    return

            self.ticket_data["contact"] = phone

            matched_customer, match_score = match_customer(
                str(self.ticket_data.get("company", "") or ""),
                phone,
                self.customer_directory,
            )
            if matched_customer:
                self.ticket_data["company"] = matched_customer["name"]
                if matched_customer.get("phone"):
                    self.ticket_data["contact"] = matched_customer["phone"]

            self.awaiting_contact = False
            self.awaiting_problem = True
            await self.say("Dziękuję. Proszę opisać problem.")
            return

        # Explicit conversation state beats LLM inference. If we just asked
        # for the problem, accept the next non-empty utterance as description.
        if self.awaiting_problem and not self.ticket_data.get("description"):
            self.ticket_data["description"] = text.strip()
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

                self.confirmation_pending = True
                self.confirmation_misses = 0
                self.awaiting_correction = False
                self.awaiting_company = False
                self.awaiting_contact = False
                self.awaiting_problem = False

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

        if self.awaiting_correction:
            llm_history.append({
                "role": "system",
                "content": (
                    "Użytkownik właśnie poprawia wcześniejsze dane. "
                    "Nowe wartości podane w tej wypowiedzi mają nadpisać odpowiednie stare pola."
                ),
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

        reply = result.get("reply") or "Dziękuję."
        ticket_update = result.get("ticket") or {}

        # Merge only non-empty values so a small model cannot erase data from
        # previous turns.
        for key, value in ticket_update.items():
            if value not in (None, "", [], {}):
                self.ticket_data[key] = value

        if self.awaiting_correction:
            self.awaiting_correction = False

        # Normalize contact phone numbers recognized with spaces, commas or dashes.
        contact_value = str(self.ticket_data.get("contact", "") or "")
        contact_digits = re.sub(r"\D", "", contact_value)
        if 9 <= len(contact_digits) <= 15:
            self.ticket_data["contact"] = contact_digits

        matched_customer, match_score = match_customer(
            str(self.ticket_data.get("company", "") or ""),
            str(self.ticket_data.get("contact", "") or ""),
            self.customer_directory,
        )
        if matched_customer:
            self.ticket_data["company"] = matched_customer["name"]
            if not self.ticket_data.get("contact") and matched_customer.get("phone"):
                self.ticket_data["contact"] = matched_customer["phone"]

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
        if company_ok and contact_ok and description_ok:
            result["done"] = True
            # Once the ticket is complete, do not trust a small LLM to produce
            # a clean final utterance; use a deterministic customer-facing line.
            reply = "Dziękuję, mam potrzebne informacje."
        else:
            result["done"] = False

        self.history.append({"role": "assistant", "content": json.dumps(result, ensure_ascii=False)})

        if result.get("done"):
            company = str(self.ticket_data.get("company", "") or "").strip() or "nie podano"
            contact = str(self.ticket_data.get("contact", "") or "").strip() or "nie podano"
            spoken_contact = speak_phone(contact) if contact != "nie podano" else contact
            description = str(self.ticket_data.get("description", "") or "").strip() or "nie podano"

            self.confirmation_pending = True
            self.confirmation_misses = 0
            self.awaiting_correction = False
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
                session.ticket_data["contact"] = caller_digits
                matched_customer, match_score = match_customer(
                    "",
                    caller_digits,
                    session.customer_directory,
                )
                if matched_customer:
                    session.ticket_data["company"] = matched_customer["name"]
                    if matched_customer.get("phone"):
                        session.ticket_data["contact"] = matched_customer["phone"]
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
