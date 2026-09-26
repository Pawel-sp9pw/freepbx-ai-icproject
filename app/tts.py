import io
import wave

import httpx
import numpy as np
from scipy.signal import resample_poly


def _pcm_to_int16(pcm: bytes, sampwidth: int) -> np.ndarray:
    if sampwidth == 2:
        return np.frombuffer(pcm, dtype="<i2").astype(np.int16, copy=False)

    if sampwidth == 1:
        x = np.frombuffer(pcm, dtype=np.uint8).astype(np.int16)
        return ((x - 128) << 8).astype(np.int16)

    if sampwidth == 3:
        b = np.frombuffer(pcm, dtype=np.uint8).reshape(-1, 3)
        x = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
        sign = x & 0x800000
        x = x - (sign << 1)
        return np.clip(x >> 8, -32768, 32767).astype(np.int16)

    if sampwidth == 4:
        x = np.frombuffer(pcm, dtype="<i4")
        return np.clip(x >> 16, -32768, 32767).astype(np.int16)

    raise ValueError(f"Nieobsługiwana szerokość próbki WAV: {sampwidth} bajtów")


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

    samples = _pcm_to_int16(pcm, sampwidth)

    if channels > 1:
        samples = samples.reshape(-1, channels).astype(np.int32).mean(axis=1).astype(np.int16)

    if rate != 8000:
        samples = resample_poly(samples.astype(np.float32), 8000, rate)
        samples = np.clip(np.rint(samples), -32768, 32767).astype(np.int16)

    return samples.astype("<i2", copy=False).tobytes()
