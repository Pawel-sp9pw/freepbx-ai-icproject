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
Pole "reply" ma być WYŁĄCZNIE gotowym zdaniem skierowanym do rozmówcy.
Nigdy nie wpisuj w "reply" instrukcji dla siebie, komentarzy typu "zadaj pytanie", "poproś o", "należy zebrać" ani opisu kolejnego kroku.
Zamiast "Zadaj pytanie o kontakt" napisz np. "Proszę podać numer telefonu kontaktowego."
Zasady zbierania danych:
- Zachowuj wszystkie wcześniej zebrane pola ticket i nie usuwaj ich w kolejnych odpowiedziach.
- Jeśli użytkownik odpowiada na pytanie o problem dowolnym niepustym zdaniem, potraktuj tę treść jako description, nawet jeśli jest krótka lub testowa.
- Jeśli jest description, ale nie ma title, utwórz krótki title na podstawie description.
- Nie wymagaj od użytkownika ponownego podawania informacji, którą już podał.
- Jeśli brakuje danych, zadaj tylko jedno krótkie pytanie i ustaw done=false.
- Ustaw done=true dopiero, gdy masz nazwę firmy, numer kontaktowy oraz description. Wszystkie trzy pola są wymagane przed potwierdzeniem i zapisaniem zgłoszenia.
- Nie poprawiaj ani nie wymyślaj danych użytkownika; zachowaj je możliwie dosłownie.
"""

async def ask_ollama(url: str, model: str, system_prompt: str, history: list[dict]):
    messages = [{"role": "system", "content": system_prompt + "\n" + SCHEMA_INSTRUCTION}]
    messages += history
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "think": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.1,
            "num_predict": 192,
        },
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


async def interpret_turn(url: str, model: str, expected: str, text: str, ticket: dict):
    system = """Jesteś klasyfikatorem jednej wypowiedzi klienta helpdesku.
Zwróć wyłącznie JSON:
{
  "intent": "company|contact|problem|confirm_yes|confirm_no|correction|unknown",
  "company": "",
  "contact": "",
  "description": ""
}
Nie wymyślaj danych. Wyciągaj tylko to, co rzeczywiście padło.
Jeśli klient potwierdza dane, intent=confirm_yes.
Jeśli zaprzecza lub chce coś poprawić, intent=confirm_no albo correction.
"""
    user = (
        f"Oczekiwany krok: {expected}\n"
        f"Już zebrane dane: {json.dumps(ticket, ensure_ascii=False)}\n"
        f"Wypowiedź klienta: {text}"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": "json",
        "think": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.0,
            "num_predict": 96,
        },
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{url.rstrip('/')}/api/chat", json=payload)
        r.raise_for_status()
        content = r.json()["message"]["content"]
    obj = json.loads(content)
    obj.setdefault("intent", "unknown")
    obj.setdefault("company", "")
    obj.setdefault("contact", "")
    obj.setdefault("description", "")
    return obj
