import uuid
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

    async def list_boards(self, project_id: str):
        """Return boards visible for a project.

        IC Project API deployments have exposed board filtering in more than one
        form over time, so try the collection endpoint first and keep a fallback
        to a project-scoped route. Only successful JSON list responses are used.
        """
        self._check_auth()
        if not project_id:
            raise RuntimeError("Brak ID projektu.")

        candidates = [
            (f"{self.base_url}/project/boards", {"pagination": 0, "project": project_id}),
            (f"{self.base_url}/project/boards", {"pagination": 0, "projectId": project_id}),
            (f"{self.base_url}/project/projects/{project_id}/boards", {"pagination": 0}),
        ]

        last_error = ""
        async with httpx.AsyncClient(timeout=20) as client:
            for url, params in candidates:
                try:
                    r = await client.get(url, headers=self.headers, params=params)
                    if not r.is_success:
                        last_error = f"HTTP {r.status_code}: {r.text[:200]}"
                        continue
                    data = r.json()
                    if isinstance(data, dict):
                        data = data.get("hydra:member") or data.get("items") or data.get("data") or []
                    if not isinstance(data, list):
                        continue

                    boards = []
                    for b in data:
                        if not isinstance(b, dict):
                            continue
                        # Defensive filtering in case the API ignores project filter.
                        bid_project = (
                            b.get("projectId")
                            or (b.get("project") or {}).get("id")
                            if isinstance(b.get("project"), dict)
                            else None
                        )
                        if bid_project and bid_project != project_id:
                            continue

                        slug = (
                            b.get("shortCode")
                            or b.get("slug")
                            or b.get("shortcode")
                            or ""
                        )
                        boards.append({
                            "id": b.get("id", ""),
                            "name": b.get("name") or slug or b.get("id", ""),
                            "slug": slug,
                        })
                    if boards:
                        return boards
                except Exception as e:
                    last_error = str(e)

        raise RuntimeError(
            "Nie udało się pobrać tablic dla projektu. "
            + (last_error or "API nie zwróciło listy tablic.")
        )

    async def resolve_board(self, board_slug: str):
        self._check_auth()
        board_slug = board_slug.strip()
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
