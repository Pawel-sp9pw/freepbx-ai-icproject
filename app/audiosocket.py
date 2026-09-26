import asyncio
import json
import logging
import re
import struct
import uuid
from collections import deque
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
                self.settings.get("stt_prompt", ""),
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
                await self.say(reply + suffix + " Dziękuję za zgłoszenie.")
                self.closed = True
                await asyncio.sleep(0.3)
                self.writer.close()
                return
            except Exception as e:
                self.final_status = "icp_error"
                self.final_error = str(e)
                log.exception("[%s] ICP create error", self.call_id)
                await self.say(
                    "Mam zebrane informacje, ale nie udało się teraz utworzyć zgłoszenia. "
                    "Proszę skontaktować się z serwisem. Do widzenia."
                )
                self.closed = True
                await asyncio.sleep(0.2)
                self.writer.close()
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
