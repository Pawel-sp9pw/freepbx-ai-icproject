from pathlib import Path
import json
from cryptography.fernet import Fernet

DATA_DIR = Path("/var/lib/freepbx-ai")
ETC_DIR = Path("/etc/freepbx-ai")
SETTINGS_FILE = DATA_DIR / "settings.json"
KEY_FILE = ETC_DIR / "secret.key"

DEFAULTS = {
    "ollama_url": "http://127.0.0.1:11434",
    "ollama_model": "qwen3:4b",
    "whisper_model": "small",
    "whisper_device": "cpu",
    "whisper_compute_type": "int8",
    "piper_url": "http://127.0.0.1:5000",
    "piper_voice": "pl_PL-mc_speech-medium",
    "icp_instance": "",
    "icp_token_enc": "",
    "icp_board_column": "",
    "icp_priority": "normal",
    "audiosocket_host": "0.0.0.0",
    "audiosocket_port": 9019,
    "greeting": "Dzień dobry. Tu automatyczny asystent serwisu. Proszę opisać problem.",
    "system_prompt": (
        "Jesteś polskim asystentem helpdesku. Rozmawiasz krótko i konkretnie. "
        "Masz zebrać: nazwę klienta lub firmy, opis problemu, zakres problemu, "
        "pilność i dane kontaktowe jeśli są potrzebne. "
        "Nie wymyślaj danych. Kiedy masz wystarczające informacje, ustaw done=true. "
        "Zawsze zwracaj wyłącznie poprawny JSON."
    ),
    "max_turns": 8,
    "silence_ms": 900,
}

def _fernet():
    ETC_DIR.mkdir(parents=True, exist_ok=True)
    if not KEY_FILE.exists():
        KEY_FILE.write_bytes(Fernet.generate_key())
        KEY_FILE.chmod(0o600)
    return Fernet(KEY_FILE.read_bytes())

def encrypt_secret(value: str) -> str:
    if not value:
        return ""
    return _fernet().encrypt(value.encode()).decode()

def decrypt_secret(value: str) -> str:
    if not value:
        return ""
    try:
        return _fernet().decrypt(value.encode()).decode()
    except Exception:
        return ""

def load_settings():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = DEFAULTS.copy()
    if SETTINGS_FILE.exists():
        try:
            data.update(json.loads(SETTINGS_FILE.read_text()))
        except Exception:
            pass
    return data

def save_settings(data):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    current = load_settings()
    current.update(data)
    SETTINGS_FILE.write_text(json.dumps(current, ensure_ascii=False, indent=2))
    SETTINGS_FILE.chmod(0o600)
    return current
