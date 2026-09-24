from pathlib import Path
import subprocess

WG_DIR = Path("/etc/wireguard")
WG_CONF = WG_DIR / "wg0.conf"

def status():
    try:
        p = subprocess.run(
            ["wg", "show", "wg0"],
            capture_output=True, text=True, timeout=5
        )
        if p.returncode == 0:
            return {"ok": True, "active": True, "output": p.stdout.strip()}
        return {"ok": True, "active": False, "output": p.stderr.strip()}
    except FileNotFoundError:
        return {"ok": False, "active": False, "output": "wireguard-tools nie jest zainstalowany"}
    except Exception as e:
        return {"ok": False, "active": False, "output": str(e)}

def apply_config(config_text: str, enable: bool = True):
    config_text = (config_text or "").strip()
    if not config_text:
        raise ValueError("Konfiguracja WireGuard jest pusta.")
    if "[Interface]" not in config_text or "[Peer]" not in config_text:
        raise ValueError("Konfiguracja musi zawierać sekcje [Interface] i [Peer].")

    WG_DIR.mkdir(parents=True, exist_ok=True)
    WG_CONF.write_text(config_text + "\n")
    WG_CONF.chmod(0o600)

    subprocess.run(["systemctl", "disable", "--now", "wg-quick@wg0"], capture_output=True, text=True)

    if enable:
        p = subprocess.run(
            ["systemctl", "enable", "--now", "wg-quick@wg0"],
            capture_output=True, text=True, timeout=20
        )
        if p.returncode != 0:
            raise RuntimeError((p.stderr or p.stdout or "Nie udało się uruchomić wg0").strip())
    return status()
