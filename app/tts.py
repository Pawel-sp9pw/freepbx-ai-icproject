import io
import wave
import audioop
import httpx

async def synthesize_pcm8k(piper_url: str, text: str, voice: str | None = None):
    payload = {"text": text}
    if voice:
        payload["voice"] = voice
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{piper_url.rstrip('/')}/synthesize", json=payload)
        r.raise_for_status()
        wav_bytes = r.content

    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        rate = w.getframerate()
        pcm = w.readframes(w.getnframes())

    if sampwidth != 2:
        pcm = audioop.lin2lin(pcm, sampwidth, 2)
    if channels == 2:
        pcm = audioop.tomono(pcm, 2, 0.5, 0.5)
    if rate != 8000:
        pcm, _ = audioop.ratecv(pcm, 2, 1, rate, 8000, None)
    return pcm
