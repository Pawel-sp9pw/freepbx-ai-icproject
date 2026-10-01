from pathlib import Path
import json
from cryptography.fernet import Fernet

DATA_DIR = Path("/var/lib/freepbx-ai")
ETC_DIR = Path("/etc/freepbx-ai")
SETTINGS_FILE = DATA_DIR / "settings.json"
KEY_FILE = ETC_DIR / "secret.key"

LEGACY_STT_PROMPT = "Rozmowa serwisowa po polsku. Nazwa firmy, numer telefonu, opis problemu."
CURRENT_STT_PROMPT = "Rozmowa telefoniczna z polskim serwisem IT. Dzwoniący podaje nazwę firmy, numer telefonu lub opis problemu."

DEFAULTS = {
    "whisper_model": "medium",
    "whisper_device": "cpu",
    "whisper_compute_type": "int8",
    "stt_workers": 2,
    "stt_prompt": "Rozmowa telefoniczna z polskim serwisem IT. Dzwoniący podaje nazwę firmy, numer telefonu lub opis problemu.",
    "stt_problem_hint": "Problem może dotyczyć e-recepty, P1, NFZ, faktur, paragonów, drukarki fiskalnej lub systemu MediQus.",
    "company_confirm_logprob": -0.55,
    "contact_auto_accept_logprob": -0.30,
    "problem_auto_accept_logprob": -0.30,
    "phone_validation_mode": "pl",
    "customer_directory": "",
    "company_alias_dictionary": "",
    "problem_dictionary": "e-recepta\ne-recepty\nP1\nNFZ\nfaktura\nparagon\ndrukarka fiskalna\nMediQus\nskaner kodów",
    "callerid_api_key": "",
    "piper_url": "http://127.0.0.1:5000",
    "piper_voice": "pl_PL-mc_speech-medium",
    "icp_instance": "",
    "icp_token_enc": "",
    "icp_project_id": "",
    "icp_project_name": "",
    "icp_board_slug": "",
    "icp_board_name": "",
    "icp_board_column": "",
    "icp_board_column_name": "",
    "icp_priority": "normal",
    "audiosocket_host": "0.0.0.0",
    "audiosocket_port": 9019,
    "greeting": "Dzień dobry. Tu automatyczny asystent serwisu. Proszę opisać problem.",
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

    # Migrate only exact legacy defaults. Custom user prompts are preserved.
    if data.get("stt_prompt") == LEGACY_STT_PROMPT:
        data["stt_prompt"] = CURRENT_STT_PROMPT

    # Main branch runs without an LLM. Ignore legacy keys that may still
    # exist in settings.json from older installations.
    data.pop("ollama_url", None)
    data.pop("ollama_model", None)
    data.pop("system_prompt", None)

    return data

def save_settings(data):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    current = load_settings()
    current.update(data)
    SETTINGS_FILE.write_text(json.dumps(current, ensure_ascii=False, indent=2))
    SETTINGS_FILE.chmod(0o600)
    return current
