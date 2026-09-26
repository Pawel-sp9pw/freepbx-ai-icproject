import time

_calls = {}
_last = {}
TTL_SECONDS = 300


def register_caller(call_id: str, caller: str):
    now = time.time()
    # Opportunistic cleanup.
    for key, item in list(_calls.items()):
        if now - item["ts"] > TTL_SECONDS:
            _calls.pop(key, None)

    item = {
        "caller": str(caller or ""),
        "ts": now,
    }
    _calls[str(call_id)] = item
    _last.clear()
    _last.update({"uuid": str(call_id), **item})


def consume_caller(call_id: str):
    item = _calls.pop(str(call_id), None)
    if not item:
        return ""
    if time.time() - item["ts"] > TTL_SECONDS:
        return ""
    return item.get("caller", "")


def last_registration():
    return dict(_last)
