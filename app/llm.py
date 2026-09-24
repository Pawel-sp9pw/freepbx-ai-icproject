import json
import httpx

SCHEMA_INSTRUCTION = r"""
Zwróć wyłącznie JSON:
{
  "reply": "krótka odpowiedź do klienta po polsku",
  "done": false,
  "ticket": {
    "company": "",
    "contact": "",
    "title": "",
    "description": "",
    "priority": "normal",
    "summary": ""
  }
}
Priorytet tylko: low, normal, high.
Jeśli brakuje danych, zadaj jedno krótkie pytanie i ustaw done=false.
Nie ustawiaj done=true, jeśli nie ma co najmniej sensownego title i description.
"""

async def ask_ollama(url: str, model: str, system_prompt: str, history: list[dict]):
    messages = [{"role": "system", "content": system_prompt + "\n" + SCHEMA_INSTRUCTION}]
    messages += history
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.2},
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{url.rstrip('/')}/api/chat", json=payload)
        r.raise_for_status()
        content = r.json()["message"]["content"]
    obj = json.loads(content)
    obj.setdefault("reply", "Dziękuję. Proszę powiedzieć coś więcej o problemie.")
    obj.setdefault("done", False)
    obj.setdefault("ticket", {})
    return obj
