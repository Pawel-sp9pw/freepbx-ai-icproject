import unittest
from unittest.mock import patch

from app.icproject import ICProjectClient


class FakeResponse:
    is_success = True

    def raise_for_status(self):
        return None

    def json(self):
        return {"id": "task-id", "shortCode": "abc123"}


class FakeAsyncClient:
    last_payload = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, headers=None, json=None):
        type(self).last_payload = json
        return FakeResponse()


class ICProjectRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_task_preserves_unmatched_caller_number(self):
        client = ICProjectClient("instance", "token", "column")
        ticket = {
            "title": "Problem testowy",
            "description": "Nie działa system",
            "company": "Nierozpoznany klient",
            "contact": "600111222",
            "caller": "792032104",
        }

        with patch("app.icproject.httpx.AsyncClient", FakeAsyncClient):
            await client.create_task(ticket)

        description = FakeAsyncClient.last_payload["description"]
        self.assertIn("Numer telefonu: 792032104", description)
        self.assertIn("Kontakt: 600111222", description)
        self.assertIn("Firma/klient: Nierozpoznany klient", description)

    async def test_uncertain_summary_is_written_to_task(self):
        client = ICProjectClient("instance", "token", "column")
        ticket = {
            "title": "Problem",
            "description": "Opis",
            "summary": "UWAGA: wymagany kontakt zwrotny.",
        }

        with patch("app.icproject.httpx.AsyncClient", FakeAsyncClient):
            await client.create_task(ticket)

        description = FakeAsyncClient.last_payload["description"]
        self.assertIn("Podsumowanie AI: UWAGA: wymagany kontakt zwrotny.", description)

    async def test_priority_is_bounded_by_backend_default_when_missing(self):
        client = ICProjectClient("instance", "token", "column")
        ticket = {"title": "Problem", "description": "Opis"}

        with patch("app.icproject.httpx.AsyncClient", FakeAsyncClient):
            await client.create_task(ticket, default_priority="normal")

        self.assertEqual(FakeAsyncClient.last_payload["priority"], "normal")


if __name__ == "__main__":
    unittest.main()
