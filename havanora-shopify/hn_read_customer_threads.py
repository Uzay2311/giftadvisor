import os
import json
import hashlib
from flask import request, jsonify

CUSTOMERS_PATH = "/home/workiqapp/mysite/havanora/customers"
THREADS_PATH = "/home/workiqapp/mysite/havanora/threads"
SHARD_PREFIX_LEN = 2


def _norm(s):
    return str(s or "").strip()


def _hash_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def deterministic_device_id(unique_id: str, shop: str = "", customer_id: str = "", customer_email: str = "") -> str:
    base = _norm(unique_id)
    if not base:
        base = "|".join([_norm(shop), _norm(customer_id), _norm(customer_email)])
    if not base:
        base = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")
    return _hash_hex(base)[:32]


def deterministic_customer_key(shop: str = "", customer_id: str = "", customer_email: str = "") -> str:
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
    return _hash_hex(key_id)[:SHARD_PREFIX_LEN]


def _customer_path(key_id: str) -> str:
    return os.path.join(CUSTOMERS_PATH, _shard_prefix(key_id), key_id, "customer.json")


def _threads_index_path(key_id: str) -> str:
    return os.path.join(THREADS_PATH, _shard_prefix(key_id), key_id, "threads_index.json")


def _read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def hn_read_customer_threads():
    """
    POST /hn_read_customer_threads
    Body:
      {
        "unique_id": "stable id if available",
        "shop": "...",
        "customer_id": "...",            # MUST be Shopify customer_id when available
        "customer_email": "...",
        "limit": 10
      }
    Reads:
      - If customer_id/customer_email present -> read by CUSTOMER KEY
      - Else -> read by DEVICE KEY
    Returns:
      {
        "key_id": "...",
        "customer_key": "..." | "",
        "device_id": "...",
        "customer": {...} | null,        # customer.customer_id will be Shopify customer_id
        "threads": [{"thread_id","thread_name","last_message_at"}...]
      }
    """
    try:
        if request.method == "OPTIONS":
            return ("", 204)

        data = request.get_json(silent=True) or {}

        unique_id = _norm(data.get("unique_id"))
        shop = _norm(data.get("shop"))
        customer_id = _norm(data.get("customer_id"))
        customer_email = _norm(data.get("customer_email"))
        limit = data.get("limit")
        try:
            limit = int(limit) if limit is not None else 10
        except Exception:
            limit = 10
        limit = max(1, min(limit, 50))

        device_id = deterministic_device_id(unique_id, shop=shop, customer_id=customer_id, customer_email=customer_email)
        customer_key = deterministic_customer_key(shop=shop, customer_id=customer_id, customer_email=customer_email)
        key_id = customer_key or device_id

        customer = _read_json(_customer_path(key_id))
        idx = _read_json(_threads_index_path(key_id))
        threads = []
        if isinstance(idx, dict) and isinstance(idx.get("threads"), list):
            for t in idx["threads"][:limit]:
                if not isinstance(t, dict):
                    continue
                threads.append({
                    "thread_id": _norm(t.get("thread_id")),
                    "thread_name": _norm(t.get("thread_name")) or _norm(t.get("thread_id")),
                    "last_message_at": _norm(t.get("last_message_at")),
                })

        # Ensure customer_id field is always Shopify customer_id (if present in request)
        if isinstance(customer, dict) and customer_id and _norm(customer.get("customer_id")) != customer_id:
            customer["customer_id"] = customer_id

        return jsonify({
            "key_id": key_id,
            "customer_key": customer_key or "",
            "device_id": device_id,
            "customer": customer if isinstance(customer, dict) else None,
            "threads": threads,
        })

    except Exception as e:
        print("Exception in hn_read_customer_threads:", repr(e))
        return jsonify({"error": "Server error"}), 500
