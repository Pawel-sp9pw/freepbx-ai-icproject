import asyncio
import logging
import base64
import secrets
import subprocess
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from .config import load_settings, save_settings, encrypt_secret, decrypt_secret
from .icproject import ICProjectClient
from .audiosocket import start_audiosocket_server
from .wireguard import status as wireguard_status, apply_config as wireguard_apply
from .monitoring import init_db, runtime_status, service_status, list_calls, get_call

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(title="FreePBX AI → IC Project")

ADMIN_PASSWORD_FILE = Path("/etc/freepbx-ai/admin-password")
UPDATE_STATE_FILE = Path("/var/lib/freepbx-ai/update.state")
UPDATE_LOG_FILE = Path("/var/lib/freepbx-ai/update.log")

@app.middleware("http")
async def basic_auth(request: Request, call_next):
    if request.url.path == "/api/health":
        return await call_next(request)

    expected_password = ""
    if ADMIN_PASSWORD_FILE.exists():
        expected_password = ADMIN_PASSWORD_FILE.read_text().strip()

    auth = request.headers.get("Authorization", "")
    valid = False
    if auth.startswith("Basic ") and expected_password:
        try:
            raw = base64.b64decode(auth[6:]).decode("utf-8")
            username, password = raw.split(":", 1)
            valid = (
                secrets.compare_digest(username, "admin")
                and secrets.compare_digest(password, expected_password)
            )
        except Exception:
            valid = False

    if not valid:
        return Response(
            content="Authentication required",
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="FreePBX AI"'},
        )
    return await call_next(request)

templates = Jinja2Templates(directory="/opt/freepbx-ai-icproject/app/templates")

@app.on_event("startup")
async def startup():
    init_db()
    asyncio.create_task(start_audiosocket_server())

@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/update/status")
async def update_status():
    state = UPDATE_STATE_FILE.read_text().strip() if UPDATE_STATE_FILE.exists() else "idle"
    log = UPDATE_LOG_FILE.read_text(errors="replace")[-12000:] if UPDATE_LOG_FILE.exists() else ""
    return {"state": state, "log": log}


@app.post("/api/update/start")
async def update_start():
    try:
        check = subprocess.run(
            ["systemctl", "is-active", "freepbx-ai-update.service"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if check.stdout.strip() == "active":
            return {"ok": False, "message": "Aktualizacja już trwa."}

        UPDATE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        UPDATE_STATE_FILE.write_text("starting\n")
        UPDATE_LOG_FILE.write_text("Uruchamianie aktualizacji...\n")

        p = subprocess.run(
            [
                "systemd-run",
                "--unit=freepbx-ai-update",
                "--collect",
                "/bin/bash",
                "/opt/freepbx-ai-icproject/scripts/update-agent.sh",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if p.returncode != 0:
            UPDATE_STATE_FILE.write_text(f"error:{p.returncode}\n")
            return {"ok": False, "message": (p.stderr or p.stdout).strip()}

        return {"ok": True, "message": "Aktualizacja uruchomiona w tle."}
    except Exception as e:
        return {"ok": False, "message": str(e)}


@app.get("/api/dashboard/status")
async def dashboard_status():
    rt = runtime_status()
    rt["services"] = service_status()
    return rt


@app.get("/api/calls")
async def calls(limit: int = 30):
    return {"items": list_calls(limit)}


@app.get("/api/calls/{call_id}")
async def call_details(call_id: str):
    item = get_call(call_id)
    if not item:
        raise HTTPException(status_code=404, detail="Rozmowa nie istnieje.")
    return item

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    s = load_settings()
    safe = s.copy()
    safe["icp_token_enc"] = ""
    safe["icp_token_set"] = bool(decrypt_secret(s.get("icp_token_enc", "")))
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "s": safe, "wg": wireguard_status()},
    )

@app.post("/save")
async def save(
    request: Request,
    ollama_url: str = Form(...),
    ollama_model: str = Form(...),
    whisper_model: str = Form(...),
    whisper_device: str = Form(...),
    whisper_compute_type: str = Form(...),
    piper_url: str = Form(...),
    piper_voice: str = Form(...),
    icp_instance: str = Form(""),
    icp_token: str = Form(""),
    icp_project_id: str = Form(""),
    icp_project_name: str = Form(""),
    icp_board_slug: str = Form(""),
    icp_board_name: str = Form(""),
    icp_board_column: str = Form(""),
    icp_board_column_name: str = Form(""),
    icp_priority: str = Form("normal"),
    greeting: str = Form(...),
    system_prompt: str = Form(...),
    max_turns: int = Form(8),
    silence_ms: int = Form(900),
):
    current = load_settings()
    data = {
        "ollama_url": ollama_url,
        "ollama_model": ollama_model,
        "whisper_model": whisper_model,
        "whisper_device": whisper_device,
        "whisper_compute_type": whisper_compute_type,
        "piper_url": piper_url,
        "piper_voice": piper_voice,
        "icp_instance": icp_instance,
        "icp_project_id": icp_project_id,
        "icp_project_name": icp_project_name,
        "icp_board_slug": icp_board_slug,
        "icp_board_name": icp_board_name,
        "icp_board_column": icp_board_column,
        "icp_board_column_name": icp_board_column_name,
        "icp_priority": icp_priority,
        "greeting": greeting,
        "system_prompt": system_prompt,
        "max_turns": max_turns,
        "silence_ms": silence_ms,
    }
    if icp_token.strip():
        data["icp_token_enc"] = encrypt_secret(icp_token.strip())
    else:
        data["icp_token_enc"] = current.get("icp_token_enc", "")
    save_settings(data)
    return RedirectResponse("/?saved=1", status_code=303)

@app.post("/api/test/icp")
async def test_icp(
    icp_instance: str = Form(""),
    icp_token: str = Form(""),
    icp_board_column: str = Form(""),
):
    s = load_settings()

    instance = icp_instance.strip() or s.get("icp_instance", "")
    token = icp_token.strip() or decrypt_secret(s.get("icp_token_enc", ""))
    board_column = icp_board_column.strip() or s.get("icp_board_column", "")

    client = ICProjectClient(instance, token, board_column)
    ok, message = await client.test()
    return {"ok": ok, "message": message}


def _icp_client_from_form(instance: str = "", token: str = "", board_column: str = ""):
    s = load_settings()
    resolved_instance = instance.strip() or s.get("icp_instance", "")
    resolved_token = token.strip() or decrypt_secret(s.get("icp_token_enc", ""))
    resolved_column = board_column.strip() or s.get("icp_board_column", "")
    return ICProjectClient(resolved_instance, resolved_token, resolved_column)


@app.post("/api/icp/projects")
async def icp_projects(
    icp_instance: str = Form(""),
    icp_token: str = Form(""),
):
    try:
        client = _icp_client_from_form(icp_instance, icp_token)
        return {"ok": True, "items": await client.list_projects()}
    except Exception as e:
        return {"ok": False, "message": str(e), "items": []}


@app.post("/api/icp/columns")
async def icp_columns(
    board_slug: str = Form(...),
    icp_instance: str = Form(""),
    icp_token: str = Form(""),
):
    try:
        client = _icp_client_from_form(icp_instance, icp_token)
        return {"ok": True, "items": await client.list_board_columns(board_slug)}
    except Exception as e:
        return {"ok": False, "message": str(e), "items": []}


@app.post("/api/test/ollama")
async def test_ollama():
    s = load_settings()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{s['ollama_url'].rstrip('/')}/api/tags")
            r.raise_for_status()
            models = [x.get("name") for x in r.json().get("models", [])]
        return {"ok": True, "models": models}
    except Exception as e:
        return {"ok": False, "message": str(e)}

@app.post("/api/test/piper")
async def test_piper():
    s = load_settings()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{s['piper_url'].rstrip('/')}/info")
            r.raise_for_status()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "message": str(e)}

@app.get("/api/wireguard/status")
async def api_wireguard_status():
    return wireguard_status()

@app.post("/api/wireguard/apply")
async def api_wireguard_apply(
    config: str = Form(...),
    enable: bool = Form(True),
):
    try:
        return wireguard_apply(config, enable)
    except Exception as e:
        return {"ok": False, "active": False, "output": str(e)}


@app.post("/api/change-password")
async def change_password(
    current_password: str = Form(...),
    new_password: str = Form(...),
    new_password_repeat: str = Form(...),
):
    if not ADMIN_PASSWORD_FILE.exists():
        return {"ok": False, "message": "Brak pliku hasła administratora."}

    stored = ADMIN_PASSWORD_FILE.read_text().strip()

    if not secrets.compare_digest(current_password, stored):
        return {"ok": False, "message": "Bieżące hasło jest nieprawidłowe."}

    if len(new_password) < 12:
        return {"ok": False, "message": "Nowe hasło musi mieć co najmniej 12 znaków."}

    if new_password != new_password_repeat:
        return {"ok": False, "message": "Nowe hasła nie są identyczne."}

    if secrets.compare_digest(new_password, stored):
        return {"ok": False, "message": "Nowe hasło musi różnić się od obecnego."}

    ADMIN_PASSWORD_FILE.write_text(new_password + "\n")
    ADMIN_PASSWORD_FILE.chmod(0o600)

    return {"ok": True, "message": "Hasło zmienione. Przy następnym żądaniu użyj nowego hasła."}
