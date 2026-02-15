import os
import json
import time
import hashlib
import tempfile
from datetime import datetime, timezone
from flask import request, jsonify

CUSTOMERS_PATH = "/home/workiqapp/mysite/havanora/customers"
THREADS_PATH = "/home/workiqapp/mysite/havanora/threads"

# Sharding tuned for 100k+ MAU: 256 buckets (2 hex chars) keeps per-dir counts low.
SHARD_PREFIX_LEN = 2


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def _atomic_write_json(path: str, obj: dict):
    _safe_mkdir(os.path.dirname(path))
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _norm(s):
    return str(s or "").strip()


def _hash_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def deterministic_device_id(unique_id: str, shop: str = "", customer_id: str = "", customer_email: str = "") -> str:
    """
    Deterministic across sessions.
    unique_id should be stable from the client; fallback to shop+customer_id/email if absent.
    """
    base = _norm(unique_id)
    if not base:
        base = "|".join([_norm(shop), _norm(customer_id), _norm(customer_email)])
    if not base:
        base = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")
    return _hash_hex(base)[:32]


def deterministic_customer_key(shop: str = "", customer_id: str = "", customer_email: str = "") -> str:
    """
    Deterministic customer-level key when a user is signed in.
    Prefers customer_id; falls back to customer_email.
    """
    s = _norm(shop)
    cid = _norm(customer_id)
    email = _norm(customer_email).lower()

    if cid:
        base = f"{s}|cid|{cid}"
    elif email:
        base = f"{s}|email|{email}"
    else:
        return ""
    return _hash_hex(base)[:32]


def _shard_prefix(key_id: str) -> str:
    h = _hash_hex(key_id)
    return h[:SHARD_PREFIX_LEN]


def _customer_dir(key_id: str) -> str:
    return os.path.join(CUSTOMERS_PATH, _shard_prefix(key_id), key_id)


def _threads_dir(key_id: str) -> str:
    return os.path.join(THREADS_PATH, _shard_prefix(key_id), key_id)


def _customer_path(key_id: str) -> str:
    return os.path.join(_customer_dir(key_id), "customer.json")


def _threads_index_path(key_id: str) -> str:
    return os.path.join(_threads_dir(key_id), "threads_index.json")


def _thread_path(key_id: str, thread_id: str) -> str:
    return os.path.join(_threads_dir(key_id), f"{thread_id}.json")


def _ensure_customer_record(key_id: str, shop: str, customer_id: str, customer_email: str, device_id: str = ""):
    path = _customer_path(key_id)
    rec = _read_json(path)
    now = _utc_now_iso()

    if not isinstance(rec, dict):
        rec = {
            # Primary key for storage/routing:
            # - signed-in: key_id == customer_key (customer-level)
            # - anonymous: key_id == device_id (device-level)
            "key_id": key_id,
            # Keep device granularity even when keyed by customer
            "device_id": _norm(device_id),
            "devices": list({d for d in [_norm(device_id)] if d}),
            "shop": _norm(shop),
            # NOTE: This field is now explicitly the Shopify customer id (if available).
            "customer_id": _norm(customer_id),
            "customer_email": _norm(customer_email),
            "created_at": now,
            "updated_at": now,
            "last_seen_at": now,
            "sessions": 0,
            # NEW: profile container (upserted from LLM when present)
            "customer_profile": None,
        }
        _atomic_write_json(path, rec)
        return rec

    # Update minimal fields if provided + keep device list
    changed = False
    if shop and rec.get("shop") != shop:
        rec["shop"] = _norm(shop)
        changed = True
    if customer_id and rec.get("customer_id") != customer_id:
        rec["customer_id"] = _norm(customer_id)
        changed = True
    if customer_email and rec.get("customer_email") != customer_email:
        rec["customer_email"] = _norm(customer_email)
        changed = True

    if device_id:
        rec["device_id"] = _norm(device_id) or rec.get("device_id") or ""
        if not isinstance(rec.get("devices"), list):
            rec["devices"] = []
        if _norm(device_id) and _norm(device_id) not in set(_norm(x) for x in rec["devices"]):
            rec["devices"].append(_norm(device_id))
        changed = True

    rec["last_seen_at"] = now
    rec["updated_at"] = now
    rec["sessions"] = int(rec.get("sessions") or 0) + 1
    changed = True

    if changed:
        _atomic_write_json(path, rec)
    return rec


def _load_threads_index(key_id: str) -> dict:
    idx_path = _threads_index_path(key_id)
    idx = _read_json(idx_path)
    if not isinstance(idx, dict):
        return {"key_id": key_id, "updated_at": _utc_now_iso(), "threads": []}
    if "threads" not in idx or not isinstance(idx["threads"], list):
        idx["threads"] = []
    return idx


def _save_threads_index(key_id: str, idx: dict):
    idx["key_id"] = key_id
    idx["updated_at"] = _utc_now_iso()
    _atomic_write_json(_threads_index_path(key_id), idx)


def _upsert_thread_meta(idx: dict, thread_id: str, thread_name: str = "", last_message_at: str = ""):
    thread_id = _norm(thread_id)
    if not thread_id:
        return idx

    now = _utc_now_iso()
    thread_name = _norm(thread_name)
    last_message_at = _norm(last_message_at) or now

    # Find existing
    threads = idx.get("threads") if isinstance(idx.get("threads"), list) else []
    found = None
    for t in threads:
        if isinstance(t, dict) and _norm(t.get("thread_id")) == thread_id:
            found = t
            break

    if found is None:
        found = {
            "thread_id": thread_id,
            "thread_name": thread_name or thread_id,
            "last_message_at": last_message_at,
            "created_at": now,
        }
        threads.append(found)
    else:
        if thread_name:
            found["thread_name"] = thread_name
        found["last_message_at"] = last_message_at

    # Sort desc by last_message_at for fast "last 10"
    def _k(x):
        try:
            return _norm(x.get("last_message_at"))
        except Exception:
            return ""

    threads.sort(key=_k, reverse=True)
    idx["threads"] = threads
    return idx


def hn_telemetry():
    """
    POST /hn_telemetry
    Body:
      {
        "unique_id": "stable id from client if available",
        "shop": "...",
        "customer_id": "...",            # MUST be Shopify customer_id when available
        "customer_email": "...",
        "event": "thread_touch|message|session|thread_meta|customer_profile|...",
        "thread_id": "YYYY-MM-DD or uuid",
        "thread_name": "display name",
        "thread_summary": "..." (optional)
        "customer_profile": {...} (optional)
        "message": {"role":"user|assistant","content":"..."}   (optional)
      }

    Storage keying:
      - If customer_id/customer_email present -> stored under CUSTOMER KEY (customer-level)
      - Else -> stored under DEVICE KEY (device-level)

    Writes:
      customers/<shard>/<key_id>/customer.json
      threads/<shard>/<key_id>/threads_index.json
      threads/<shard>/<key_id>/<thread_id>.json (optional append / meta update)
    """
    try:
        if request.method == "OPTIONS":
            return ("", 204)

        data = request.get_json(silent=True) or {}

        unique_id = _norm(data.get("unique_id"))
        shop = _norm(data.get("shop"))
        customer_id = _norm(data.get("customer_id"))
        customer_email = _norm(data.get("customer_email"))

        # Always compute device_id from unique_id (keeps device-level granularity)
        device_id = deterministic_device_id(unique_id, shop=shop, customer_id=customer_id, customer_email=customer_email)

        # If user is signed in, route all writes by customer_key; otherwise by device_id
        customer_key = deterministic_customer_key(shop=shop, customer_id=customer_id, customer_email=customer_email)
        key_id = customer_key or device_id

        customer_rec = _ensure_customer_record(key_id, shop, customer_id, customer_email, device_id=device_id)

        event = _norm(data.get("event")) or "session"
        thread_id = _norm(data.get("thread_id"))
        thread_name = _norm(data.get("thread_name"))
        thread_summary = data.get("thread_summary")
        if not isinstance(thread_summary, str):
            thread_summary = ""
        customer_profile = data.get("customer_profile") if isinstance(data.get("customer_profile"), dict) else None

        msg = data.get("message") if isinstance(data.get("message"), dict) else None

        now = _utc_now_iso()

        # upsert customer_profile if present
        if customer_profile is not None:
            customer_rec["customer_profile"] = customer_profile
            customer_rec["updated_at"] = now
            _atomic_write_json(_customer_path(key_id), customer_rec)

        touched_thread = False
        if thread_id:
            idx = _load_threads_index(key_id)
            idx = _upsert_thread_meta(idx, thread_id, thread_name=thread_name, last_message_at=now)
            _save_threads_index(key_id, idx)
            touched_thread = True

            # Thread doc update (meta + optional append)
            tpath = _thread_path(key_id, thread_id)
            tdoc = _read_json(tpath)
            if not isinstance(tdoc, dict):
                tdoc = {"thread_id": thread_id, "key_id": key_id, "device_id": device_id, "created_at": now, "messages": []}
            if "messages" not in tdoc or not isinstance(tdoc["messages"], list):
                tdoc["messages"] = []

            # Always keep latest device_id on the thread doc too
            tdoc["device_id"] = device_id

            if thread_name:
                tdoc["thread_name"] = thread_name

            if thread_summary:
                tdoc["thread_summary"] = thread_summary

            if msg and _norm(msg.get("role")) and _norm(msg.get("content")):
                # Append to thread file (kept as a list of messages)
                tdoc["messages"].append({
                    "role": _norm(msg.get("role")),
                    "content": _norm(msg.get("content")),
                    "ts": now,
                    "event": event,
                    "device_id": device_id,
                })

            tdoc["updated_at"] = now
            _atomic_write_json(tpath, tdoc)

        return jsonify({
            "ok": True,
            "key_id": key_id,
            "customer_key": customer_key or "",
            "device_id": device_id,
            "customer": customer_rec,
            "touched_thread": touched_thread,
            "ts": now,
        })

    except Exception as e:
        print("Exception in hn_telemetry:", repr(e))
        return jsonify({"error": "Server error"}), 500
