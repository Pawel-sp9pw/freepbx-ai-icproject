import asyncio
import json
import logging
import re
import struct
import uuid
from collections import deque
from difflib import SequenceMatcher
import webrtcvad

from .config import load_settings, decrypt_secret
from .stt import transcribe_pcm16
from .tts import synthesize_pcm8k
from .llm import ask_ollama
from .icproject import ICProjectClient
from .monitoring import call_started, add_message, finish_call

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
    writer.write(bytes([typ]) + len(payload).to_bytes(2, "big") + payload)
    await writer.drain()

async def send_pcm(writer, pcm: bytes):
    # 20 ms @ 8 kHz, mono, 16-bit = 320 bytes
    for i in range(0, len(pcm), 320):
        chunk = pcm[i:i+320]
        if len(chunk) < 320:
            chunk += b"\x00" * (320 - len(chunk))
        await send_packet(writer, TYPE_PCM_8K, chunk)
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
        self.stt_misses = 0
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

    async def start(self):
        await self.say(self.settings["greeting"])

    async def handle_pcm(self, payload: bytes):
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
                if silence_ms >= int(self.settings.get("silence_ms", 900)):
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
            # Do not speak on every noise/silence fragment. Ask for repetition
            # only after two consecutive failed recognitions.
            if self.stt_misses >= 2:
                self.stt_misses = 0
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
            yes_phrases = ("tak", "zgadza się", "zgadza sie", "potwierdzam", "poprawnie", "wszystko się zgadza", "wszystko sie zgadza")
            no_phrases = ("nie", "nie zgadza", "popraw", "błąd", "blad", "zmień", "zmien")

            if any(p in normalized for p in yes_phrases):
                self.confirmation_pending = False
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
                    return
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
                    return

            if any(p in normalized for p in no_phrases):
                self.confirmation_pending = False
                self.awaiting_correction = True
                await self.say(
                    "Dobrze. Proszę podać ponownie tylko dane, które mam poprawić: "
                    "nazwę firmy, numer kontaktowy albo opis problemu."
                )
                return

            await self.say("Proszę odpowiedzieć: tak, jeśli dane są poprawne, albo nie, jeśli mam je poprawić.")
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

        reply = result.get("reply") or "Dziękuję."
        ticket_update = result.get("ticket") or {}

        # Merge only non-empty values so a small model cannot erase data from
        # previous turns.
        for key, value in ticket_update.items():
            if value not in (None, "", [], {}):
                self.ticket_data[key] = value

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
        description_ok = bool(str(self.ticket_data.get("description", "")).strip())
        if company_ok and description_ok:
            result["done"] = True
            # Once the ticket is complete, do not trust a small LLM to produce
            # a clean final utterance; use a deterministic customer-facing line.
            reply = "Dziękuję, mam potrzebne informacje."

        self.history.append({"role": "assistant", "content": json.dumps(result, ensure_ascii=False)})

        if result.get("done"):
            company = str(self.ticket_data.get("company", "") or "").strip() or "nie podano"
            contact = str(self.ticket_data.get("contact", "") or "").strip() or "nie podano"
            description = str(self.ticket_data.get("description", "") or "").strip() or "nie podano"

            self.confirmation_pending = True
            self.awaiting_correction = False
            await self.say(
                "Podsumuję zgłoszenie. "
                f"Firma: {company}. "
                f"Numer kontaktowy: {contact}. "
                f"Problem: {description}. "
                "Czy dane są poprawne? Proszę powiedzieć tak lub nie."
            )
            return

        if self.turns >= int(self.settings.get("max_turns", 8)):
            await self.say(
                "Nie udało się zebrać kompletu informacji. Proszę skontaktować się z serwisem."
            )
            self.final_status = "incomplete"
            self.closed = True
            self.writer.close()
            return

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
