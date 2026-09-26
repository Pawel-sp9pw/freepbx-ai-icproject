import uuid
from datetime import datetime, timezone, timedelta
import httpx


class ICProjectClient:
    def __init__(self, instance: str, token: str, board_column: str = ""):
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
            "User-Agent": "FreePBX-AI-ICProject/0.2",
        }

    def _check_auth(self):
        if not self.instance or not self.token:
            raise RuntimeError("Brak instance slug lub tokenu IC Project.")

    async def test(self):
        if not self.instance or not self.token:
            return False, "Brak instance slug lub tokenu."
        url = f"{self.base_url}/project/projects?pagination=0"
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url, headers=self.headers)
            if r.is_success:
                return True, f"OK ({r.status_code})"
            return False, f"HTTP {r.status_code}: {r.text[:300]}"

    async def list_projects(self):
        self._check_auth()
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(
                f"{self.base_url}/project/projects?pagination=0",
                headers=self.headers,
            )
            r.raise_for_status()
            data = r.json()
        return [
            {
                "id": p.get("id", ""),
                "name": p.get("name") or p.get("shortCode") or p.get("id", ""),
                "shortCode": p.get("shortCode", ""),
            }
            for p in data
            if p.get("id")
        ]

    async def resolve_board(self, board_slug: str):
        self._check_auth()
        board_slug = board_slug.strip().rstrip("/")
        if board_slug.startswith("http://") or board_slug.startswith("https://"):
            board_slug = board_slug.split("/")[-1]
        if not board_slug:
            raise RuntimeError("Brak identyfikatora tablicy.")

        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(
                f"{self.base_url}/project/boards/s/{board_slug}/get-kanban-board",
                headers=self.headers,
            )
            if not r.is_success:
                raise RuntimeError(f"IC Project HTTP {r.status_code}: {r.text[:300]}")
            data = r.json()

        if not data.get("id"):
            raise RuntimeError("API nie zwróciło ID tablicy.")
        return data

    async def list_board_columns(self, board_slug: str):
        board = await self.resolve_board(board_slug)
        board_id = board["id"]

        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(
                f"{self.base_url}/project/boards/{board_id}/board-columns",
                headers=self.headers,
            )
            if not r.is_success:
                raise RuntimeError(f"IC Project HTTP {r.status_code}: {r.text[:300]}")
            data = r.json()

        return [
            {
                "id": c.get("id", ""),
                "name": c.get("name") or c.get("id", ""),
            }
            for c in data
            if isinstance(c, dict) and c.get("id")
        ]

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

        now = datetime.now(timezone.utc)
        payload = {
            "identifier": str(uuid.uuid4()),
            "boardColumn": self.board_column,
            "name": name,
            "description": description[:12000],
            "dateStart": now.isoformat(),
            "dateEnd": (now + timedelta(days=1)).isoformat(),
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
