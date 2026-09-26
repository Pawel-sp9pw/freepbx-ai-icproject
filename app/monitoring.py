import sqlite3
import subprocess
import threading
import psutil
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("/var/lib/freepbx-ai/calls.db")
_lock = threading.Lock()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _lock, _db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS calls (
                call_id TEXT PRIMARY KEY,
                peer_ip TEXT,
                peer_port INTEGER,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                ticket_ref TEXT,
                error TEXT
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                call_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                FOREIGN KEY(call_id) REFERENCES calls(call_id)
            );

            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS resource_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                cpu_percent REAL NOT NULL,
                ram_percent REAL NOT NULL,
                disk_percent REAL NOT NULL,
                load_1 REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_resource_samples_created_at
                ON resource_samples(created_at);
            """
        )
        # If the process was killed/restarted, old "active" rows are no longer active.
        conn.execute(
            "UPDATE calls SET status='interrupted', ended_at=? WHERE status='active'",
            (_now(),),
        )


def _set_meta(conn, key, value):
    conn.execute(
        "INSERT INTO meta(key,value) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def call_started(call_id: str, peer):
    peer_ip = ""
    peer_port = None
    if isinstance(peer, (tuple, list)) and peer:
        peer_ip = str(peer[0])
        if len(peer) > 1:
            try:
                peer_port = int(peer[1])
            except Exception:
                peer_port = None
    elif peer:
        peer_ip = str(peer)

    now = _now()
    with _lock, _db() as conn:
        conn.execute(
            """
            INSERT INTO calls(call_id,peer_ip,peer_port,started_at,status)
            VALUES(?,?,?,?, 'active')
            ON CONFLICT(call_id) DO UPDATE SET
                peer_ip=excluded.peer_ip,
                peer_port=excluded.peer_port,
                started_at=excluded.started_at,
                ended_at=NULL,
                status='active',
                ticket_ref=NULL,
                error=NULL
            """,
            (call_id, peer_ip, peer_port, now),
        )
        _set_meta(conn, "freepbx_last_seen", now)
        _set_meta(conn, "freepbx_last_peer", peer_ip)


def add_message(call_id: str, role: str, content: str):
    if not content:
        return
    with _lock, _db() as conn:
        conn.execute(
            "INSERT INTO messages(call_id,created_at,role,content) VALUES(?,?,?,?)",
            (call_id, _now(), role, str(content)),
        )


def finish_call(call_id: str, status="ended", ticket_ref="", error=""):
    with _lock, _db() as conn:
        conn.execute(
            """
            UPDATE calls
            SET ended_at=?, status=?, ticket_ref=?, error=?
            WHERE call_id=?
            """,
            (_now(), status, ticket_ref or None, error or None, call_id),
        )


def list_calls(limit=30):
    limit = max(1, min(int(limit), 200))
    with _db() as conn:
        rows = conn.execute(
            """
            SELECT c.*,
                   ROUND((julianday(COALESCE(c.ended_at, CURRENT_TIMESTAMP)) -
                          julianday(c.started_at)) * 86400) AS duration_seconds,
                   (SELECT COUNT(*) FROM messages m WHERE m.call_id=c.call_id) AS message_count
            FROM calls c
            ORDER BY c.started_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_call(call_id: str):
    with _db() as conn:
        row = conn.execute("SELECT * FROM calls WHERE call_id=?", (call_id,)).fetchone()
        if not row:
            return None
        messages = conn.execute(
            "SELECT created_at,role,content FROM messages WHERE call_id=? ORDER BY id",
            (call_id,),
        ).fetchall()
    result = dict(row)
    result["messages"] = [dict(m) for m in messages]
    return result


def runtime_status():
    with _db() as conn:
        active = conn.execute(
            "SELECT call_id,peer_ip,started_at FROM calls WHERE status='active' ORDER BY started_at"
        ).fetchall()
        meta = {
            r["key"]: r["value"]
            for r in conn.execute(
                "SELECT key,value FROM meta WHERE key IN ('freepbx_last_seen','freepbx_last_peer')"
            ).fetchall()
        }

    return {
        "active_calls": [dict(r) for r in active],
        "active_count": len(active),
        "freepbx_last_seen": meta.get("freepbx_last_seen"),
        "freepbx_last_peer": meta.get("freepbx_last_peer"),
    }


def service_status():
    services = [
        ("freepbx-ai", "Agent"),
        ("ollama", "Ollama"),
        ("piper-ai", "Piper TTS"),
        ("caddy", "Caddy"),
        ("wg-quick@wg0", "WireGuard"),
    ]
    result = []
    for unit, label in services:
        try:
            p = subprocess.run(
                ["systemctl", "is-active", unit],
                capture_output=True,
                text=True,
                timeout=3,
            )
            state = (p.stdout or p.stderr).strip() or "unknown"
            result.append({
                "unit": unit,
                "label": label,
                "state": state,
                "ok": state == "active",
            })
        except Exception as e:
            result.append({
                "unit": unit,
                "label": label,
                "state": "error",
                "ok": False,
                "error": str(e),
            })
    return result


def linux_resource_status():
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    swap = psutil.swap_memory()
    boot = datetime.fromtimestamp(psutil.boot_time(), tz=timezone.utc)
    uptime_seconds = max(0, int((datetime.now(timezone.utc) - boot).total_seconds()))
    load1, load5, load15 = psutil.getloadavg()
    cpu_count = psutil.cpu_count(logical=True) or 1

    result = {
        "cpu_percent": round(psutil.cpu_percent(interval=0.15), 1),
        "cpu_count": cpu_count,
        "load_1": round(load1, 2),
        "load_5": round(load5, 2),
        "load_15": round(load15, 2),
        "load_1_percent": round(min(100.0, (load1 / cpu_count) * 100.0), 1),
        "ram_percent": round(vm.percent, 1),
        "ram_used": vm.used,
        "ram_total": vm.total,
        "swap_percent": round(swap.percent, 1),
        "swap_used": swap.used,
        "swap_total": swap.total,
        "disk_percent": round(disk.percent, 1),
        "disk_used": disk.used,
        "disk_total": disk.total,
        "uptime_seconds": uptime_seconds,
        "temperature_c": _temperature_status(),
        "processes": process_resource_status(),
    }
    _record_resource_sample(result)
    return result


def _temperature_status():
    try:
        temps = psutil.sensors_temperatures(fahrenheit=False) or {}
        values = []
        for entries in temps.values():
            for entry in entries:
                if entry.current is not None:
                    values.append(float(entry.current))
        if values:
            return round(max(values), 1)
    except Exception:
        pass
    return None


def process_resource_status():
    groups = {
        "Agent": {"match": ("uvicorn", "app.main:app", "freepbx-ai"), "cpu": 0.0, "rss": 0, "count": 0},
        "Ollama": {"match": ("ollama",), "cpu": 0.0, "rss": 0, "count": 0},
        "Piper TTS": {"match": ("piper.http_server", "piper"), "cpu": 0.0, "rss": 0, "count": 0},
        "Caddy": {"match": ("caddy",), "cpu": 0.0, "rss": 0, "count": 0},
    }

    candidates = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(proc.info.get("cmdline") or [])
            name = proc.info.get("name") or ""
            haystack = f"{name} {cmd}".lower()
            for label, group in groups.items():
                if any(token.lower() in haystack for token in group["match"]):
                    candidates.append((label, proc))
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    for label, proc in candidates:
        try:
            cpu = proc.cpu_percent(interval=0.03)
            mem = proc.memory_info().rss
            groups[label]["cpu"] += cpu
            groups[label]["rss"] += mem
            groups[label]["count"] += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    result = []
    for label, group in groups.items():
        result.append({
            "label": label,
            "cpu_percent": round(group["cpu"], 1),
            "rss": int(group["rss"]),
            "count": group["count"],
        })
    return result


def _record_resource_sample(resource):
    now = datetime.now(timezone.utc)
    with _lock, _db() as conn:
        row = conn.execute(
            "SELECT value FROM meta WHERE key='last_resource_sample'"
        ).fetchone()
        should_write = True
        if row and row["value"]:
            try:
                previous = datetime.fromisoformat(row["value"])
                should_write = (now - previous).total_seconds() >= 30
            except Exception:
                should_write = True

        if should_write:
            conn.execute(
                """
                INSERT INTO resource_samples(created_at,cpu_percent,ram_percent,disk_percent,load_1)
                VALUES(?,?,?,?,?)
                """,
                (
                    now.isoformat(),
                    float(resource["cpu_percent"]),
                    float(resource["ram_percent"]),
                    float(resource["disk_percent"]),
                    float(resource["load_1"]),
                ),
            )
            _set_meta(conn, "last_resource_sample", now.isoformat())
            cutoff = (now.timestamp() - 7 * 86400)
            cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()
            conn.execute(
                "DELETE FROM resource_samples WHERE created_at < ?",
                (cutoff_iso,),
            )


def resource_history(hours=24):
    hours = max(1, min(int(hours), 168))
    cutoff = datetime.fromtimestamp(
        datetime.now(timezone.utc).timestamp() - hours * 3600,
        tz=timezone.utc,
    ).isoformat()
    with _db() as conn:
        rows = conn.execute(
            """
            SELECT created_at,cpu_percent,ram_percent,disk_percent,load_1
            FROM resource_samples
            WHERE created_at >= ?
            ORDER BY created_at
            """,
            (cutoff,),
        ).fetchall()

    items = [dict(r) for r in rows]
    if len(items) > 360:
        step = max(1, len(items) // 360)
        items = items[::step]
    return items
