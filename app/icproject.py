import uuid
import httpx

class ICProjectClient:
    def __init__(self, instance: str, token: str, board_column: str):
        self.instance = instance.strip()
        self.token = token.strip()
        self.board_column = board_column.strip()

    @property
    def base_url(self):
        return f"https://app.icproject.com/api/instance/{self.instance}"

    @property
    def headers(self):
        return {
            "X-Auth-Token": self.token,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "FreePBX-AI-ICProject/0.1",
        }

    async def test(self):
        if not self.instance or not self.token:
            return False, "Brak instance slug lub tokenu."
        url = f"{self.base_url}/project/projects?pagination=0"
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url, headers=self.headers)
            if r.is_success:
                return True, f"OK ({r.status_code})"
            return False, f"HTTP {r.status_code}: {r.text[:300]}"

    async def create_task(self, ticket: dict, default_priority="normal"):
        if not self.board_column:
            raise RuntimeError("Brak ID kolumny IC Project.")
        name = (ticket.get("title") or "Zgłoszenie telefoniczne")[:150]
        description = ticket.get("description") or ""
        caller = ticket.get("caller") or ""
        company = ticket.get("company") or ""
        contact = ticket.get("contact") or ""

        extra = []
        if company:
            extra.append(f"Firma/klient: {company}")
        if caller:
            extra.append(f"Numer telefonu: {caller}")
        if contact:
            extra.append(f"Kontakt: {contact}")
        if ticket.get("summary"):
            extra.append(f"Podsumowanie AI: {ticket['summary']}")
        if extra:
            description = description + "\n\n" + "\n".join(extra)

        payload = {
            "identifier": str(uuid.uuid4()),
            "boardColumn": self.board_column,
            "name": name,
            "description": description[:12000],
            "priority": ticket.get("priority") or default_priority,
        }

        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(
                f"{self.base_url}/project/tasks",
                headers=self.headers,
                json=payload,
            )
            if not r.is_success:
                raise RuntimeError(f"IC Project HTTP {r.status_code}: {r.text[:500]}")
            return r.json()
