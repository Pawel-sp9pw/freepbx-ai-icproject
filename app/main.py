import asyncio
import logging
import base64
import secrets
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from .config import load_settings, save_settings, encrypt_secret, decrypt_secret
from .icproject import ICProjectClient
from .audiosocket import start_audiosocket_server
from .wireguard import status as wireguard_status, apply_config as wireguard_apply

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(title="FreePBX AI → IC Project")

ADMIN_PASSWORD_FILE = Path("/etc/freepbx-ai/admin-password")

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
    asyncio.create_task(start_audiosocket_server())

@app.get("/api/health")
async def health():
    return {"status": "ok"}

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
    icp_board_column: str = Form(""),
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
        "icp_board_column": icp_board_column,
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
async def test_icp():
    s = load_settings()
    client = ICProjectClient(
        s.get("icp_instance", ""),
        decrypt_secret(s.get("icp_token_enc", "")),
        s.get("icp_board_column", ""),
    )
    ok, message = await client.test()
    return {"ok": ok, "message": message}

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
