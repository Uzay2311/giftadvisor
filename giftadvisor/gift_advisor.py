"""
Gift Advisor AI - Helps users find the perfect gift for specific occasions.
Adapted from havanora-shopify hn_chat_bot.py, focused on gift recommendations.
Includes web search for Amazon products; LLM picks best 3 from top 10 results.
"""
from dotenv import load_dotenv

load_dotenv()

from flask import request, jsonify, Response, stream_with_context
from openai import OpenAI
from datetime import datetime, timezone
from typing import Optional
import os
import json
import logging
import re
import time
import hashlib
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    import psycopg
except Exception:
    psycopg = None

logger = logging.getLogger("gift_advisor")

from product_search import scrape_amazon_searches

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
try:
    PRODUCTS_PER_SEARCH = max(10, min(int(os.getenv("PRODUCTS_PER_SEARCH", "20")), 30))
except Exception:
    PRODUCTS_PER_SEARCH = 20

ABUSE_DB_PATH = os.path.join(os.path.dirname(__file__), "logs", "abuse_flags.sqlite3")
GENERIC_ABUSE_REPLY = "I can help with gift suggestions, but I cannot process this request right now. Please try again with a normal, concise gift-related question."
TELEMETRY_RETENTION_DAYS = max(1, min(int(os.getenv("TELEMETRY_RETENTION_DAYS", "90")), 3650))
TELEMETRY_DB_URL = os.getenv("DATABASE_URL", "").strip()
_telemetry_init_lock = threading.Lock()
_telemetry_db_ready = False
_telemetry_last_cleanup_ts = 0.0

# Prefer Responses API if available; fallback to Chat Completions
def _call_llm(messages: list, stream: bool = False):
    """Call LLM - tries Responses API first, then Chat Completions."""
    # Convert to Chat Completions format (role: system/user/assistant)
    chat_messages = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "developer":
            role = "system"
        if role in ("system", "user", "assistant") and content:
            chat_messages.append({"role": role, "content": content})

    return client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        messages=chat_messages,
        temperature=0.2,
        response_format={"type": "json_object"},
        stream=stream,
    )

def merge_search_state(old: Optional[dict], new: dict) -> dict:
    """Merge new constraints into existing search state."""
    merged = old.copy() if isinstance(old, dict) else {}

    for k, v in new.items():
        if v is None:
            continue
        if isinstance(v, list) and not v:
            continue
        merged[k] = v

    return merged


def _recipient_gender_from_message(message: str) -> tuple[Optional[str], Optional[str]]:
    m = str(message or "").strip().lower()
    if not m:
        return (None, None)
    mapping = [
        (("wife", "girlfriend", "mom", "mother", "sister", "daughter"), "women"),
        (("husband", "boyfriend", "dad", "father", "brother", "son"), "men"),
    ]
    for words, gender in mapping:
        for w in words:
            if re.search(rf"\b{re.escape(w)}\b", m):
                return (w, gender)
    return (None, None)


def _normalize_people_profiles(raw_profiles) -> dict:
    out = {"active_profile_id": None, "profiles": []}
    if not isinstance(raw_profiles, dict):
        return out
    profiles = raw_profiles.get("profiles")
    active_id = str(raw_profiles.get("active_profile_id") or "").strip()
    if not isinstance(profiles, list):
        return out
    seen = set()
    for p in profiles[:20]:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id") or "").strip()
        if not pid or pid in seen:
            continue
        seen.add(pid)
        ctx = p.get("context") if isinstance(p.get("context"), dict) else {}
        label = str(p.get("label") or "").strip()
        if not label:
            label = str(ctx.get("recipient") or pid).strip()
        out["profiles"].append({"id": pid, "label": label, "context": ctx})
    if out["profiles"]:
        ids = {p["id"] for p in out["profiles"]}
        out["active_profile_id"] = active_id if active_id in ids else out["profiles"][0]["id"]
    return out


def _next_profile_id(people_profiles: dict) -> str:
    ids = {str(p.get("id")) for p in people_profiles.get("profiles") or [] if isinstance(p, dict)}
    n = 1
    while True:
        pid = f"p{n}"
        if pid not in ids:
            return pid
        n += 1


def _build_effective_input_context(
    people_profiles: dict,
    active_profile_id: Optional[str],
    gift_context: Optional[dict],
    message: str,
    occasion: str,
    budget_min,
    budget_max,
) -> tuple[dict, str, dict]:
    store = _normalize_people_profiles(people_profiles)
    profiles = store["profiles"]
    msg_recipient, msg_gender = _recipient_gender_from_message(message)

    chosen_idx = None
    if msg_recipient:
        for i, p in enumerate(profiles):
            pr = str(((p.get("context") or {}).get("recipient") or "")).strip().lower()
            if pr and pr == msg_recipient:
                chosen_idx = i
                break
        if chosen_idx is None:
            pid = _next_profile_id(store)
            new_ctx = {"recipient": msg_recipient}
            if msg_gender:
                new_ctx["gender"] = msg_gender
            profiles.append({"id": pid, "label": msg_recipient, "context": new_ctx})
            chosen_idx = len(profiles) - 1
    else:
        current = str(active_profile_id or store.get("active_profile_id") or "").strip()
        if current:
            for i, p in enumerate(profiles):
                if str(p.get("id") or "") == current:
                    chosen_idx = i
                    break
        if chosen_idx is None and profiles:
            chosen_idx = 0
        if chosen_idx is None:
            pid = _next_profile_id(store)
            profiles.append({"id": pid, "label": "gift profile", "context": {}})
            chosen_idx = 0

    profile = profiles[chosen_idx]
    selected_id = str(profile.get("id") or _next_profile_id(store))
    base = profile.get("context") if isinstance(profile.get("context"), dict) else {}
    incoming_ctx = gift_context if isinstance(gift_context, dict) else {}
    if msg_recipient:
        # Explicit recipient mention means we are pivoting person context.
        # Keep the selected profile as source-of-truth and only carry generic
        # cross-profile constraints that are safe to reuse.
        carry_keys = ("occasion", "budget_min", "budget_max")
        carry_ctx = {k: incoming_ctx.get(k) for k in carry_keys if incoming_ctx.get(k) is not None}
        base = merge_search_state(base, carry_ctx)
    else:
        base = merge_search_state(base, incoming_ctx)

    ui_ctx = {}
    if occasion:
        ui_ctx["occasion"] = occasion
    try:
        if budget_min is not None and str(budget_min).strip() != "":
            ui_ctx["budget_min"] = int(float(budget_min))
        if budget_max is not None and str(budget_max).strip() != "":
            ui_ctx["budget_max"] = int(float(budget_max))
    except Exception:
        pass
    if msg_recipient:
        ui_ctx["recipient"] = msg_recipient
    if msg_gender:
        ui_ctx["gender"] = msg_gender

    effective = merge_search_state(base, ui_ctx)
    profile["context"] = effective
    profile["label"] = str(effective.get("recipient") or profile.get("label") or selected_id)
    store["active_profile_id"] = selected_id
    return effective, selected_id, store


def _update_active_profile_context(people_profiles: dict, active_profile_id: str, context: dict) -> dict:
    store = _normalize_people_profiles(people_profiles)
    if not active_profile_id:
        return store
    for p in store["profiles"]:
        if str(p.get("id") or "") == str(active_profile_id):
            p["context"] = merge_search_state(p.get("context") if isinstance(p.get("context"), dict) else {}, context or {})
            p["label"] = str((p["context"].get("recipient") or p.get("label") or active_profile_id))
            break
    return store


GIFT_ADVISOR_SYSTEM_PROMPT = """
You are a warm, knowledgeable Gift Advisor—like a gift expert in a boutique. Your goal is to help people find the perfect gift.

You are NOT a general chat companion. You focus exclusively on gift-giving advice.

Core behaviors:
- FIRST gather information: occasion, budget, recipient's interests. Do NOT search until you have enough context.
- Ask about the occasion (birthday, anniversary, wedding, holiday, graduation, etc.)
- Learn about the recipient: age or age range, interests, relationship to giver, personality
- Understand budget constraints and preferences
- Only after you have occasion + budget + (interests or product), suggest products
- Keep responses concise and actionable (2-4 sentences typically)
- Use bullet points for multiple suggestions
- Be encouraging and creative—help people feel confident in their choice
- Never end a turn with a dead-end statement. End with one clear next step (a focused question or a concrete action choice).
- If user sends social/filler acknowledgments (e.g., "nice products thanks"), respond warmly and continue the flow without unnecessary re-search.
- Conversation goal: collect enough useful detail -> generate high-quality options -> revise with feedback -> narrow down to one best gift.

Formatting:
- Use \\n for new lines. Avoid \\n\\n.
- Prefer bullets for lists.
- Keep suggestions skimmable and scannable.
""".strip()

GIFT_ADVISOR_ENRICHMENT = """
Return structured JSON only. The "reply" field must contain your actual conversational response to the user—never a placeholder or schema description.
Do not include markdown fences, explanations, or any text before/after the JSON object.

{
  "reply": "Your actual reply text here",
  "gift_context": {
    "product": "string | null",
    "color": "string | null",
    "gender": "men | women | unisex | null",
    "recipient": "string | null",
    "budget_min": number | null,
    "budget_max": number | null,
    "occasion": "string | null",
    "interests": ["string"],
    "liked_products": [{"title": "string", "price": "string"}],
    "disliked_products": [{"title": "string", "price": "string"}]
  },
  "search_strategy": {
    "mode": "specific | explore",
    "queries": [
      {
        "query": "string (Amazon search string)",
        "subtitle": "string (user-friendly display title, e.g. 'Thoughtful picks for her' or 'Gifts for her 45th'—NOT the raw query)"
      }
    ]
  } | null,
  "shortlist_intent": boolean
}

RULES (VERY IMPORTANT):

0. "reply" must be your real response (e.g. "I'd love to help! What gift are you looking for?"). NEVER output schema text like "string (UI-ready...)" or placeholders.
1. You MUST build on the FULL conversation history.
2. NEVER drop hard constraints once stated:
   - product
   - color
   - gender (wife → women, husband → men)
   - budget
3. When the user revises (adds info), UPDATE the existing constraints.
4. search_strategy.mode:
   - "specific" → 1 query (clear product + constraints)
   - "explore" → 3 queries (same hard constraints, different angles)
5. Queries MUST be Amazon-style search strings. When budget is set (budget_min/budget_max), append it to each query—e.g. "women's pickleball shoes under $50", "red running shoes between $30 and $80", "gift for wife below $100".
6. subtitle MUST be a short, user-friendly display title (e.g. "Thoughtful picks for her", "Gifts for her 45th"). NEVER use the raw search query as subtitle.
7. reply MUST NEVER include search queries.
8. NEVER search when gathering info. If your reply ASKS for occasion, budget, or interests, return search_strategy = null. No products—just the question. One question per turn.
9. Only return search_strategy (and trigger a search) when you have occasion + budget + (interests/hobbies or explicit product). Until then, ask for what's missing and return null. Do NOT search and ask in the same turn.
10. When the user has provided enough (occasion, budget, interests/product), return search_strategy with queries. Do NOT return null when you have enough context.
11. Treat values already present in gift_context (including UI-selected occasion/budget) as known facts; do not ask for them again.
12. Mention on-screen occasion/budget options ONLY when either occasion or budget is still missing. If occasion and budget are already known in gift_context, do NOT mention on-screen options.
13. Ask for recipient age (or age range) as part of exploration. If the recipient is a child/teen (e.g., son, daughter, kid, teenager, 13-17), collect age before any product search.
14. If user asks for "more options", "other ideas", "something different", or is unhappy with current picks, keep ALL hard constraints but generate NEW creative query angles. Do not repeat prior queries from previous_search_queries/people_profiles context unless no alternatives exist.
15. For "more options" requests with enough context, prefer search_strategy.mode="explore" with 3 diverse, non-overlapping query intents (e.g., accessories vs premium core product vs experiential gift) while preserving recipient, age fit, occasion, and budget.
16. If user says "surprise me" (or similar open-ended intent), you may use a trending/explore approach and return diverse creative queries even when interests are limited. Still preserve known hard constraints (recipient, age fit, occasion, budget) and avoid repeating prior queries.
17. If the user is asking to compare/rank/evaluate already shown products (e.g., "which one is best", "which should I pick", "compare these"), set search_strategy = null and set shortlist_intent = true so the system routes to shortlist analysis (no new external search).
18. If the user explicitly asks to explore/trending/surprise (e.g. "help me explore", "show me trending"), you may proceed with search_strategy even when interests are missing, as long as recipient + occasion + budget are already known.
19. If gift_context includes liked_products or disliked_products, use them as preference signals: align next queries with liked product patterns and avoid themes/categories similar to disliked products.
20. If the user message is social acknowledgment/filler (e.g., "nice products thanks", "awesome, thank you"), keep the conversation warm and engaging but set search_strategy = null unless they explicitly ask for more/refinement.
21. Avoid dead-end filler lines (e.g., "This will help me...") without a next step. If you are not searching in this turn, ask exactly one focused follow-up question.
22. Maintain this flow naturally: discovery -> search -> revise -> pinpoint one best gift. After showing options, guide user toward either refinement or selecting a single best pick.
23. When enough context is already known, avoid repetitive clarification loops; move the conversation forward with a concrete action choice.
24. If the user switches recipient/person (e.g., "for my wife", "let's move to my son", "that was for my son"), pivot immediately to that person. Do NOT carry prior person's interests/product as confirmed facts; ask one focused discovery question for the new person unless that profile already has enough context.
25. Do NOT claim curated products, direct seller links, or specific store listings unless you are also returning search_strategy to trigger real product retrieval. If search_strategy is null, keep the reply conversational and ask the next focused question/action.
26. Avoid vague filler follow-ups like "Could you share a bit more detail?". Ask a concrete, context-aware next question that references known recipient/product constraints.
""".strip()

PRODUCT_RANKER_PROMPT = """
You are a product ranking assistant for gift recommendations.

Task:
- Select the best top_k products from candidate products for the given context.
- Prioritize: recipient fit, occasion fit, budget fit, interest/brand fit, and reliability signals.
- If context contains liked_products/disliked_products, prefer items similar to liked ones and penalize items similar to disliked ones.
- Reliability signals to prefer: higher rating, higher reviews, and stronger bought_last_month.
- Avoid repetition: do NOT select near-duplicate products that are very similar in title/description/specs (e.g., same model with only minor variant differences) unless unique options are unavailable.
- Treat the same underlying listing as a duplicate even when title formatting differs (prefix/suffix changes, reordered words, minor punctuation differences, or repeated model names).
- Deprioritize bulk/commercial listings (e.g., multi-pack classroom sets, wholesale packs, institutional bundles, large quantity lots) unless the user explicitly asks for bulk quantities.
- Reject obvious mismatches (e.g., infant/kids products when recipient is an adult wife/woman) unless explicitly requested.
- Prefer category diversity when multiple query categories are present; avoid selecting all products from one category unless clearly superior.
- Do not invent products.
- Return ONLY valid JSON:
{
  "selected_ids": ["p1", "p4", "p2", "p7", "p3"],
  "reasons": ["short reason 1", "short reason 2", "short reason 3", "short reason 4", "short reason 5"]
}
""".strip()

SHORTLIST_QA_PROMPT = """
You are a gift advisor assistant answering follow-up questions about products already shortlisted in this chat.

Rules:
- Use ONLY the shortlisted products provided by the user payload.
- Do NOT propose new searches or new products.
- If the answer cannot be determined from shortlist data, say so clearly.
- Keep the answer concise, practical, and user-facing.
- Respect gift_context.liked_products and gift_context.disliked_products when deciding "best"; prefer similarity to liked and avoid disliked patterns.
- Output plain text only in "reply". No markdown links, no image markdown, no raw URLs.
- Prefer this structure:
  - One-line recommendation first (which one is best and why).
  - Then up to 3 short bullet lines with: product name, price, rating/reviews, and one reason.
  - End with one short next-step question.
- Return ONLY valid JSON:
{
  "reply": "string"
}
""".strip()


def _abuse_db_connect():
    os.makedirs(os.path.dirname(ABUSE_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(ABUSE_DB_PATH, timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _init_abuse_db():
    try:
        with _abuse_db_connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS device_abuse (
                    device_id TEXT PRIMARY KEY,
                    system_abuser_flag INTEGER NOT NULL DEFAULT 0,
                    abuse_score INTEGER NOT NULL DEFAULT 0,
                    window_start INTEGER NOT NULL DEFAULT 0,
                    req_count INTEGER NOT NULL DEFAULT 0,
                    last_seen INTEGER NOT NULL DEFAULT 0,
                    last_msg_hash TEXT,
                    repeat_count INTEGER NOT NULL DEFAULT 0,
                    updated_at INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_device_abuse_flag ON device_abuse(system_abuser_flag)"
            )
    except Exception as e:
        logger.warning("Failed to initialize abuse DB: %s", e)


def _normalize_device_id(raw_device_id: Optional[str]) -> str:
    rid = str(raw_device_id or "").strip().lower()
    if rid and re.fullmatch(r"[a-z0-9._-]{8,128}", rid):
        return rid
    fallback_src = (
        (request.headers.get("X-Forwarded-For") or request.remote_addr or "unknown")
        + "|"
        + (request.headers.get("User-Agent") or "ua")
    )
    return "anon-" + hashlib.sha256(fallback_src.encode("utf-8")).hexdigest()[:24]


def _normalize_session_id(raw_session_id: Optional[str], device_id: str) -> str:
    sid = str(raw_session_id or "").strip().lower()
    if sid and re.fullmatch(r"[a-z0-9._:-]{8,128}", sid):
        return sid
    seed = f"{device_id}:{int(time.time() // 1800)}"
    return "sess-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]


def _telemetry_enabled() -> bool:
    return bool(TELEMETRY_DB_URL) and psycopg is not None


def _ensure_telemetry_table():
    global _telemetry_db_ready
    if _telemetry_db_ready or not _telemetry_enabled():
        return
    with _telemetry_init_lock:
        if _telemetry_db_ready:
            return
        try:
            with psycopg.connect(TELEMETRY_DB_URL, connect_timeout=3) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS telemetry_events (
                            id BIGSERIAL PRIMARY KEY,
                            ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            event_name TEXT NOT NULL,
                            device_id TEXT NOT NULL,
                            session_id TEXT NOT NULL,
                            route TEXT NOT NULL DEFAULT '/gift_advisor',
                            latency_ms INTEGER,
                            props JSONB NOT NULL DEFAULT '{}'::jsonb
                        );
                        """
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS telemetry_events_ts_idx ON telemetry_events (ts DESC);"
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS telemetry_events_event_ts_idx ON telemetry_events (event_name, ts DESC);"
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS telemetry_events_device_ts_idx ON telemetry_events (device_id, ts DESC);"
                    )
                conn.commit()
            _telemetry_db_ready = True
        except Exception as e:
            logger.warning("Telemetry init skipped: %s", e)


def _telemetry_cleanup_if_needed():
    global _telemetry_last_cleanup_ts
    if not _telemetry_enabled() or not _telemetry_db_ready:
        return
    now = time.time()
    if now - _telemetry_last_cleanup_ts < 3600:
        return
    _telemetry_last_cleanup_ts = now
    try:
        with psycopg.connect(TELEMETRY_DB_URL, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM telemetry_events WHERE ts < NOW() - (%s || ' days')::interval",
                    (str(TELEMETRY_RETENTION_DAYS),),
                )
            conn.commit()
    except Exception as e:
        logger.warning("Telemetry cleanup failed: %s", e)


def _track_telemetry_event(
    event_name: str,
    device_id: str,
    session_id: str,
    latency_ms: Optional[int] = None,
    props: Optional[dict] = None,
):
    if not _telemetry_enabled():
        return
    _ensure_telemetry_table()
    if not _telemetry_db_ready:
        return
    safe_props = props if isinstance(props, dict) else {}
    try:
        props_json = json.dumps(safe_props, ensure_ascii=False)
    except Exception:
        props_json = "{}"
    try:
        with psycopg.connect(TELEMETRY_DB_URL, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO telemetry_events (event_name, device_id, session_id, route, latency_ms, props)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        str(event_name or "").strip() or "unknown_event",
                        str(device_id or "unknown_device"),
                        str(session_id or "unknown_session"),
                        "/gift_advisor",
                        int(latency_ms) if isinstance(latency_ms, (int, float)) else None,
                        props_json,
                    ),
                )
            conn.commit()
    except Exception as e:
        logger.warning("Telemetry insert failed: %s", e)
        return
    _telemetry_cleanup_if_needed()


def _score_message_risk(message: str) -> int:
    m = str(message or "").strip()
    if not m:
        return 0
    score = 0
    if len(m) > 1200:
        score += 3
    elif len(m) > 700:
        score += 2
    if re.search(r"(https?://|www\.)", m, re.I):
        score += 2
    if re.search(r"(buy now|cheap followers|click here|free money|casino|crypto giveaway)", m, re.I):
        score += 3
    if re.search(r"(.)\1{9,}", m):
        score += 2
    if m.count("\n") > 20:
        score += 1
    return score


def _check_and_update_abuse(device_id: str, message: str) -> int:
    now = int(time.time())
    msg_hash = hashlib.sha256(str(message or "").strip().lower().encode("utf-8")).hexdigest()
    try:
        with _abuse_db_connect() as conn:
            row = conn.execute(
                """
                SELECT system_abuser_flag, abuse_score, window_start, req_count, last_msg_hash, repeat_count
                FROM device_abuse WHERE device_id=?
                """,
                (device_id,),
            ).fetchone()

            if row is None:
                system_abuser_flag = 0
                abuse_score = 0
                window_start = now
                req_count = 0
                last_msg_hash = None
                repeat_count = 0
            else:
                (
                    system_abuser_flag,
                    abuse_score,
                    window_start,
                    req_count,
                    last_msg_hash,
                    repeat_count,
                ) = row

            if system_abuser_flag == 1:
                conn.execute(
                    "UPDATE device_abuse SET last_seen=?, updated_at=? WHERE device_id=?",
                    (now, now, device_id),
                )
                return 1

            if now - int(window_start or 0) > 60:
                window_start = now
                req_count = 0

            req_count = int(req_count or 0) + 1
            risk_score = _score_message_risk(message)

            if last_msg_hash and msg_hash == last_msg_hash:
                repeat_count = int(repeat_count or 0) + 1
            else:
                repeat_count = 1

            if req_count >= 25:
                risk_score += 3
            if repeat_count >= 6:
                risk_score += 3

            # Soft decay to avoid permanent escalation on occasional bursts.
            abuse_score = max(0, int(abuse_score or 0) - 1) + risk_score
            system_abuser_flag = 1 if abuse_score >= 12 else 0

            conn.execute(
                """
                INSERT INTO device_abuse
                (device_id, system_abuser_flag, abuse_score, window_start, req_count, last_seen, last_msg_hash, repeat_count, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(device_id) DO UPDATE SET
                    system_abuser_flag=excluded.system_abuser_flag,
                    abuse_score=excluded.abuse_score,
                    window_start=excluded.window_start,
                    req_count=excluded.req_count,
                    last_seen=excluded.last_seen,
                    last_msg_hash=excluded.last_msg_hash,
                    repeat_count=excluded.repeat_count,
                    updated_at=excluded.updated_at
                """,
                (
                    device_id,
                    int(system_abuser_flag),
                    int(abuse_score),
                    int(window_start),
                    int(req_count),
                    now,
                    msg_hash,
                    int(repeat_count),
                    now,
                ),
            )
            return int(system_abuser_flag)
    except Exception as e:
        logger.warning("Abuse check failed, allowing request: %s", e)
        return 0




def _normalize_history(history, max_turns=16):
    if not isinstance(history, list):
        return []
    out = []
    for h in history:
        role = str(h.get("role", "")).strip().lower()
        content = str(h.get("content", "")).strip()
        if not content or role not in ("user", "assistant"):
            continue
        out.append({"role": role, "content": content})
    return out[-max_turns:]


def _canonical_product_key(title: str, link: str) -> str:
    """Build a stable key for deduping the same listing across title variants."""
    t = re.sub(r"\s+", " ", str(title or "").strip().lower())
    l = str(link or "").strip().lower()
    if l:
        m = re.search(r"/(?:dp|gp/product)/([a-z0-9]{10})(?:[/?]|$)", l)
        if m:
            return f"asin:{m.group(1)}"
        m = re.search(r"[?&]asin=([a-z0-9]{10})(?:[&#]|$)", l)
        if m:
            return f"asin:{m.group(1)}"
        base = re.sub(r"^https?://", "", l)
        base = re.sub(r"[?#].*$", "", base).rstrip("/")
        if base:
            return f"link:{base}"
    if t:
        norm_t = re.sub(r"[^a-z0-9]+", " ", t)
        norm_t = re.sub(r"\s{2,}", " ", norm_t).strip()
        return f"title:{norm_t}"
    return ""


def _flatten_product_candidates(raw_results: list, max_candidates: int = 1000) -> list:
    """Flatten and dedupe products from all query result buckets."""
    if not isinstance(raw_results, list):
        return []
    out = []
    seen = set()
    idx = 1
    for row in raw_results:
        query = (row.get("query") or "").strip() if isinstance(row, dict) else ""
        products = row.get("products") if isinstance(row, dict) else []
        for p in products or []:
            if not isinstance(p, dict):
                continue
            title = (p.get("title") or "").strip()
            link = (p.get("link") or "").strip()
            key = _canonical_product_key(title, link)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append({
                "id": f"p{idx}",
                "query": query,
                "title": title,
                "description": (p.get("description") or "").strip(),
                "price": (p.get("price") or "").strip(),
                "rating": p.get("rating"),
                "reviews": p.get("reviews"),
                "bought_last_month": p.get("bought_last_month"),
                "product": p,
            })
            idx += 1
            if len(out) >= max_candidates:
                return out
    return out


def _rank_products_with_llm(
    candidates: list,
    message: str,
    history: list,
    gift_context: Optional[dict],
    query_subtitles: list,
    top_k: int = 5,
) -> list:
    """Use a second LLM call to rank candidates and return top products."""
    if not isinstance(candidates, list) or not candidates:
        return []

    ranking_view = []
    for c in candidates:
        ranking_view.append({
            "id": c.get("id"),
            "query": c.get("query"),
            "title": c.get("title"),
            "description": c.get("description"),
            "price": c.get("price"),
            "rating": c.get("rating"),
            "reviews": c.get("reviews"),
            "bought_last_month": c.get("bought_last_month"),
        })

    user_payload = {
        "user_message": message,
        "gift_context": gift_context or {},
        "recent_history": history[-8:] if isinstance(history, list) else [],
        "query_hints": [{"query": q, "subtitle": s} for q, s in (query_subtitles or [])[:3]],
        "candidates": ranking_view,
        "top_k": int(top_k),
    }

    rank_messages = [
        {"role": "developer", "content": PRODUCT_RANKER_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]

    try:
        resp = _call_llm(rank_messages, stream=False)
        raw = (resp.choices[0].message.content or "").strip()
        parsed = _extract_first_json_object(raw)
        selected_ids = parsed.get("selected_ids") if isinstance(parsed, dict) else None
        if not isinstance(selected_ids, list):
            selected_ids = []
    except Exception as e:
        logger.warning("Product ranker LLM failed, using fallback top %s: %s", top_k, e)
        selected_ids = []

    by_id = {c.get("id"): c for c in candidates if c.get("id")}
    chosen_candidates = []
    used = set()

    for pid in selected_ids:
        if not isinstance(pid, str):
            continue
        c = by_id.get(pid)
        if not c:
            continue
        key = _canonical_product_key(
            c.get("title"),
            ((c.get("product") or {}).get("link") if isinstance(c.get("product"), dict) else "") or "",
        )
        if key and key in used:
            continue
        if key:
            used.add(key)
        chosen_candidates.append(c)
        if len(chosen_candidates) >= top_k:
            break

    if len(chosen_candidates) < top_k:
        for c in candidates:
            key = _canonical_product_key(
                c.get("title"),
                ((c.get("product") or {}).get("link") if isinstance(c.get("product"), dict) else "") or "",
            )
            if key and key in used:
                continue
            if key:
                used.add(key)
            chosen_candidates.append(c)
            if len(chosen_candidates) >= top_k:
                break

    # Light diversity guard: when pool has 2+ query categories,
    # try to ensure top picks are not all from one query bucket.
    pool_queries = {str(c.get("query") or "").strip() for c in candidates if str(c.get("query") or "").strip()}
    picked_queries = {str(c.get("query") or "").strip() for c in chosen_candidates if str(c.get("query") or "").strip()}
    if len(pool_queries) >= 2 and len(picked_queries) < 2 and len(chosen_candidates) >= 2:
        used_titles = {
            _canonical_product_key(
                c.get("title"),
                ((c.get("product") or {}).get("link") if isinstance(c.get("product"), dict) else "") or "",
            )
            for c in chosen_candidates
        }
        used_titles.discard("")
        missing_queries = [q for q in pool_queries if q not in picked_queries]
        replacement = None
        for mq in missing_queries:
            for c in candidates:
                if str(c.get("query") or "").strip() != mq:
                    continue
                t = _canonical_product_key(
                    c.get("title"),
                    ((c.get("product") or {}).get("link") if isinstance(c.get("product"), dict) else "") or "",
                )
                if t and t in used_titles:
                    continue
                replacement = c
                break
            if replacement:
                break
        if replacement:
            chosen_candidates[-1] = replacement

    return [(c.get("product") or {}) for c in chosen_candidates[:top_k]]


def _friendly_section_title(query: str, subtitle: str, idx: int) -> str:
    s = (subtitle or "").strip()
    if s:
        return s
    q = (query or "").strip()
    if q:
        import re
        # Strip budget fragments and noisy symbols, then title-case a concise phrase.
        q = re.sub(r"\b(?:under|below|over|above)\s*\$?\s*\d+(?:[.,]\d+)?\b", "", q, flags=re.I)
        q = re.sub(r"\bbetween\s*\$?\s*\d+(?:[.,]\d+)?\s*(?:-|to|and)\s*\$?\s*\d+(?:[.,]\d+)?\b", "", q, flags=re.I)
        q = re.sub(r"[^\w\s'-]", " ", q)
        q = re.sub(r"\s{2,}", " ", q).strip()
        if q:
            words = q.split()[:6]
            return " ".join(words).title()
    defaults = [
        "Top style picks",
        "Best-fit gift ideas",
        "Great options to consider",
    ]
    return defaults[(max(1, idx) - 1) % len(defaults)]


def _rank_products_by_sections(
    raw_results: list,
    query_subtitles: list,
    message: str,
    history: list,
    gift_context: Optional[dict],
) -> list:
    """Rank products per query bucket and return UI sections."""
    if not isinstance(raw_results, list) or not raw_results:
        return []

    subtitle_by_query = {q: s for q, s in (query_subtitles or []) if q}
    ordered_queries = [q for q, _ in (query_subtitles or []) if q][:3]
    if not ordered_queries:
        ordered_queries = [str(r.get("query") or "").strip() for r in raw_results[:3] if str(r.get("query") or "").strip()]

    multi_query = len(ordered_queries) > 1
    top_k = 3 if multi_query else 5
    global_candidates = _flatten_product_candidates(raw_results, max_candidates=1000)
    tasks = []
    for idx, q in enumerate(ordered_queries, start=1):
        row = next((r for r in raw_results if str(r.get("query") or "").strip() == q), None)
        row_candidates = _flatten_product_candidates([row], max_candidates=1000) if isinstance(row, dict) else []
        # Rank with full pool visibility, but keep section intent via query_hints.
        # Start with row-specific candidates and append remaining global ones.
        candidates = []
        seen = set()
        for c in row_candidates + global_candidates:
            cid = str(c.get("id") or "")
            key = _canonical_product_key(
                c.get("title"),
                ((c.get("product") or {}).get("link") if isinstance(c.get("product"), dict) else "") or "",
            )
            key = cid or key
            if not key or key in seen:
                continue
            seen.add(key)
            candidates.append(c)
        tasks.append((idx, q, row_candidates, candidates))

    def _rank_section_task(idx, q, row_candidates, candidates):
        if not candidates:
            return (idx, None)
        ranked = _rank_products_with_llm(
            candidates=candidates,
            message=message,
            history=history,
            gift_context=gift_context,
            query_subtitles=[(q, subtitle_by_query.get(q, ""))],
            top_k=top_k,
        )
        # Enforce strict section integrity: keep only products that belong to this query bucket.
        row_keys = set()
        for rc in row_candidates:
            key = _canonical_product_key(
                rc.get("title"),
                ((rc.get("product") or {}).get("link") if isinstance(rc.get("product"), dict) else "") or "",
            )
            if key:
                row_keys.add(key)
        strict_ranked = []
        used_keys = set()
        for p in ranked or []:
            key = _canonical_product_key(p.get("title"), p.get("link"))
            in_row = key in row_keys
            if not in_row:
                continue
            if key in used_keys:
                continue
            used_keys.add(key)
            strict_ranked.append(p)
            if len(strict_ranked) >= top_k:
                break
        # Refill from the same query bucket if ranker picked out-of-bucket items.
        if len(strict_ranked) < top_k:
            for rc in row_candidates:
                p = rc.get("product") or {}
                key = _canonical_product_key(p.get("title"), p.get("link"))
                if key in used_keys:
                    continue
                used_keys.add(key)
                strict_ranked.append(p)
                if len(strict_ranked) >= top_k:
                    break
        if not strict_ranked:
            return (idx, None)
        return (
            idx,
            {
                "query": f"category_{idx}",
                "subtitle": _friendly_section_title(q, subtitle_by_query.get(q, ""), idx),
                "products": strict_ranked[:top_k],
            },
        )

    sections_by_idx = {}
    max_workers = min(4, max(1, len(tasks)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_rank_section_task, idx, q, row_candidates, candidates): idx
            for idx, q, row_candidates, candidates in tasks
        }
        for fut in as_completed(futures):
            idx, section = fut.result()
            if section:
                sections_by_idx[idx] = section

    ordered = [sections_by_idx[i] for i in sorted(sections_by_idx.keys())]
    # Final guard: no duplicate listing across sections.
    seen_global = set()
    for section in ordered:
        products = section.get("products") if isinstance(section, dict) else []
        if not isinstance(products, list):
            continue
        unique_products = []
        for p in products:
            if not isinstance(p, dict):
                continue
            key = _canonical_product_key(p.get("title"), p.get("link"))
            if not key or key in seen_global:
                continue
            seen_global.add(key)
            unique_products.append(p)
        section["products"] = unique_products
    return [s for s in ordered if (s.get("products") if isinstance(s, dict) else [])]


def _safe_json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None


def _extract_first_json_object(text: str):
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        obj = _safe_json_loads(t)
        if isinstance(obj, dict):
            return obj
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                cand = t[start : i + 1]
                obj = _safe_json_loads(cand)
                if isinstance(obj, dict):
                    return obj
                return None
    return None


def _parse_search_queries(sq_list) -> list:
    """Parse search_queries into [(query, subtitle), ...]. Handles string or object format."""
    if not isinstance(sq_list, list) or not sq_list:
        return []
    out = []
    for item in sq_list[:3]:
        if isinstance(item, dict):
            q = (item.get("query") or "").strip()
            s = (item.get("subtitle") or "").strip()
        else:
            q = str(item or "").strip()
            s = ""
        if q:
            out.append((q, s))
    return out


def _extract_search_queries_from_reply(reply: str) -> list:
    """Fallback: extract product-type phrases from reply when LLM omits search_queries."""
    import re
    queries = []
    skip = ("e.g", "etc", "i.e", "what is", "who is", "could you", "please provide", "budget range", "your budget")
    def _should_skip(p):
        lower = p.lower()
        return any(s in lower for s in skip) or lower.endswith("?")
    for m in re.finditer(r'\*{2}([^*]+)\*{2}', reply or ""):
        phrase = m.group(1).strip()
        if len(phrase) > 2 and len(phrase) < 50 and not _should_skip(phrase):
            queries.append(phrase)
    for m in re.finditer(r'^[\s]*[-•*]\s+\*{0,2}([^:*\n]+)', reply or "", re.MULTILINE):
        phrase = m.group(1).strip().strip("*")
        if len(phrase) > 2 and len(phrase) < 50 and not _should_skip(phrase):
            clean = phrase.split(":")[0].strip()
            if clean and clean.lower() not in [q.lower() for q in queries]:
                queries.append(clean)
    return list(dict.fromkeys(queries))[:3]  # dedupe, max 3


def _derive_queries_from_gift_context(
    gift_context: Optional[dict],
    message: Optional[str] = None,
) -> list:
    """Derive search queries from gift_context (and message) when LLM omits them.
    Builds Amazon-style queries from product, color, gender, interests.
    """
    if not isinstance(gift_context, dict):
        return []
    product = (gift_context.get("product") or "").strip()
    color = (gift_context.get("color") or "").strip()
    gender = (gift_context.get("gender") or "").strip().lower()
    interests = [s for s in (gift_context.get("interests") or []) if isinstance(s, str) and len(s) > 1]
    queries = []
    # Infer gender from message if not in context
    if not gender and message:
        import re
        if re.search(r"\bwife\b", message, re.I):
            gender = "women"
        elif re.search(r"\bhusband\b", message, re.I):
            gender = "men"
    # Infer interests from message if not in context (e.g. "for pickleball", "for tennis")
    if not interests and message:
        import re
        for m in re.finditer(r"(?:for|use\s+(?:this|it)\s+for|into)\s+(\w+)", message, re.I):
            word = m.group(1).lower()
            if len(word) > 3 and word not in ("this", "that", "gift", "occasion"):
                interests = [word]
                break
    # Infer product from interest if user says "for pickleball" etc. (e.g. shoes + pickleball -> pickleball shoes)
    base_product = product
    if not base_product and interests:
        base_product = interests[0] + " gift"
    if interests and product and "shoes" in product.lower():
        for i in interests:
            if i.lower() in ("pickleball", "tennis", "basketball", "volleyball", "golf", "running"):
                base_product = f"{i} shoes"
                break
    if not base_product:
        for s in (gift_context.get("suggestions_so_far") or [])[:3]:
            if isinstance(s, str) and len(s) > 2:
                clean = s.strip().strip("*").split(":")[0].strip()
                if clean and clean.lower() not in [q.lower() for q in queries]:
                    queries.append(clean)
        for i in interests[:2]:
            if i.lower() not in [q.lower() for q in queries]:
                queries.append(i + " gift")
        return queries[:3]
    # Build product-based queries
    gender_prefix = "women's " if gender == "women" else "men's " if gender == "men" else ""
    color_prefix = (color + " ") if color else ""
    q1 = f"{gender_prefix}{color_prefix}{base_product}".strip()
    if q1 and q1.lower() not in [x.lower() for x in queries]:
        queries.append(q1)
    if len(queries) < 3 and color and color.lower() not in q1.lower():
        q2 = f"{color} {base_product} {gender}".strip() if gender else f"{color} {base_product}".strip()
        if q2 and q2.lower() not in [x.lower() for x in queries]:
            queries.append(q2)
    if len(queries) < 3 and gender and gender not in q1.lower():
        q3 = f"{base_product} {gender}".strip()
        if q3 and q3.lower() not in [x.lower() for x in queries]:
            queries.append(q3)
    return queries[:3]


def _user_message_to_search_queries(message: str) -> list:
    """Last-resort fallback: derive search query from user's message."""
    import re
    msg = (message or "").strip()
    if len(msg) < 5:
        return []
    vague = re.match(r"^(show\s+(me\s+)?more|other\s+options?|more\s+ideas?|anything\s+else)\s*[?!.]?$", msg, re.I)
    if vague:
        return []
    # "gift for my wife" / "gift for husband" → clean Amazon-style queries
    m = re.search(r"gift\s+(?:for\s+)?(?:my\s+)?(wife|husband|mom|mother|dad|father|girlfriend|boyfriend|friend)", msg, re.I)
    if m:
        who = m.group(1).lower()
        q = "gift for wife" if who == "wife" else "gift for husband" if who == "husband" else f"gift for {who}"
        return [q]
    stop = {"for", "my", "the", "a", "an", "to", "and", "or", "with", "around", "about", "budget", "under", "over", "no", "special", "occasion", "just", "because"}
    words = re.sub(r"[^\w\s]", " ", msg).split()
    kept = [w for w in words if w.lower() not in stop and not w.isdigit()]
    if len(kept) < 2:
        return []
    q = " ".join(kept[:8])
    q = re.sub(r"\bwife\b", "women", q, flags=re.I)
    q = re.sub(r"\bhusband\b", "men", q, flags=re.I)
    return [q] if q else []


def _compact_ui_text(reply: str) -> str:
    reply = (reply or "").strip()
    reply = reply.replace("\r\n", "\n").replace("\r", "\n")
    while "\n\n" in reply:
        reply = reply.replace("\n\n", "\n")
    return reply


def _sanitize_reply_markdown_artifacts(reply: str) -> str:
    """Strip markdown artifacts that render poorly in chat bubbles."""
    t = str(reply or "").strip()
    if not t:
        return t
    # Remove markdown images completely.
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)
    # Convert markdown links to visible link text only.
    t = re.sub(r"\[([^\]]+)\]\((?:https?://[^)]+)\)", r"\1", t)
    # Remove naked URLs that clutter response formatting.
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n[ \t]+\n", "\n\n", t)
    return _compact_ui_text(t)


def _strip_search_queries_from_reply(reply: str) -> str:
    """Remove 'Search Queries:' and everything after from reply."""
    import re
    t = (reply or "").strip()
    m = re.search(r'(?i)\*{0,2}\s*Search\s+Queries\s*\*{0,2}\s*:?\s*', t)
    if m:
        t = t[: m.start()].rstrip()
    return _compact_ui_text(t)


def _avoid_reasking_known_fields(reply: str, gift_context: Optional[dict]) -> str:
    """Remove redundant clarifying asks for fields that are already known."""
    if not isinstance(reply, str):
        return reply
    if not isinstance(gift_context, dict):
        return reply

    import re
    original = _compact_ui_text(reply or "")
    text = (reply or "").strip()
    if not text:
        return text

    has_budget = gift_context.get("budget_min") is not None or gift_context.get("budget_max") is not None

    # If occasion is already known, strip only the occasion-ask fragments.
    if gift_context.get("occasion"):
        patterns = [
            r"(?i)\b(?:could\s+you(?:\s+please)?|can\s+you|please)\s+(?:tell|share)(?:\s+me)?(?:\s+what(?:'s|\s+is))?\s+(?:the\s+)?occasion[^?.!]*[?.!]?",
            r"(?i)\bwhat(?:'s|\s+is)\s+(?:the\s+)?occasion[^?.!]*[?.!]?",
            r"(?i)\bwhat\s+occasion\s+are\s+you\s+shopping\s+for[^?.!]*[?.!]?",
            r"(?i)\bis\s+it[\s,]*(?:for\s+)?(?:a\s+)?(?:birthday|anniversary|wedding|holiday|graduation|baby\s*shower|housewarming|thank\s*you)[^?.!]*[?.!]?",
            r"(?i)\b(?:for\s+)?(?:her|your\s+wife'?s?)\s+(?:birthday|anniversary|wedding|holiday|graduation|baby\s*shower|housewarming|thank\s*you)\??",
            r"(?i)\b(?:occasion\s+for\s+the\s+gift)\b",
        ]
        for pat in patterns:
            text = re.sub(pat, " ", text)

    # If budget is already known, strip budget-ask fragments.
    if has_budget:
        budget_patterns = [
            r"(?i)\b(?:also\s*,?\s*)?what(?:'s|\s+is)\s+(?:your\s+)?budget(?:\s+range)?[^?.!]*[?.!]?",
            r"(?i)\bdo\s+you\s+have\s+(?:a\s+)?budget(?:\s+range)?[^?.!]*[?.!]?",
            r"(?i)\bcould\s+you(?:\s+please)?\s+(?:share|tell)\s+(?:me\s+)?(?:your\s+)?budget(?:\s+range)?[^?.!]*[?.!]?",
            r"(?i)\bplease\s+(?:share|tell)\s+(?:me\s+)?(?:your\s+)?budget(?:\s+range)?[^?.!]*[?.!]?",
            r"(?i)\bif\s+you\s+have\s+(?:a\s+)?specific\s+budget(?:\s+in\s+mind)?[^?.!]*[?.!]?",
            r"(?i)\bfeel\s+free\s+to\s+share\s+(?:that\s+as\s+well|your\s+budget(?:\s+range)?)\b[^?.!]*[?.!]?",
            r"(?i)\b(?:let\s+me\s+know|share|tell\s+me)\s+(?:if\s+you\s+have\s+)?(?:a\s+)?budget(?:\s+range)?\b[^?.!]*[?.!]?",
            r"(?i)\byou\s+mentioned[^?.!]*\$\s*\d+[^?.!]*is\s+that\s+correct[^?.!]*[?.!]?",
            r"(?i)\bare\s+you\s+still\s+comfortable\s+with[^?.!]*\$\s*\d+[^?.!]*[?.!]?",
            r"(?i)\byou\s+mentioned\s+(?:a\s+)?range\s+of\s+\d+\s*(?:to|-)\s*\d+[^?.!]*[?.!]?",
        ]
        for pat in budget_patterns:
            text = re.sub(pat, " ", text)
        # Catch-all: remove any remaining short sentence that asks about budget.
        text = re.sub(r"(?i)(?:^|[.!?]\s+)[^.!?]{0,140}\bbudget\b[^.!?]*[.!?]?", " ", text)

    # Remove generic UI-option boilerplate that can appear by itself.
    text = re.sub(
        r"(?i)(?:^|[.!?]\s+)you\s+can\s+also\s+tap\s+the\s+on-?screen\s+options[^.!?]*[.!?]?",
        " ",
        text,
    )
    # Remove low-value generic filler lines that often cause dead-end turns.
    text = re.sub(
        r"(?i)(?:^|[.!?]\s+)(?:thanks\s+for\s+sharing|this\s+will\s+help\s+me\s+find\s+the\s+perfect\s+options?\s+for\s+(?:him|her|them|you))[^.!?]*[.!?]?",
        " ",
        text,
    )

    # Cleanup connective artifacts after fragment removal.
    text = re.sub(r"(?i)\b(and|also)\s*(?:,)?\s*(and|also)\b", r"\1", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,;.-")
    text = re.sub(r"(?i)\b(?:and|also)\s*$", "", text).strip(" ,;.-")
    # Remove clipped trailing stubs left after aggressive phrase stripping.
    text = re.sub(r"(?i)\b(?:i\s+see|i\s+can\s+see|got\s+it|understood)\s*$", "", text).strip(" ,;.-")

    # If cleanup left a plain statement without terminal punctuation, close it cleanly.
    if text and text[-1] not in ".!?":
        text = text + "."

    cleaned = _compact_ui_text(text)
    # If cleanup became too aggressive and removed meaning, keep LLM's original wording.
    return cleaned or original


def _ensure_next_step(reply: str, gift_context: Optional[dict], has_products: bool = False) -> str:
    """Minimal safety net: prefer LLM wording, only guard empty output."""
    text = _compact_ui_text(reply or "")
    if not text:
        return "I'd love to help—what would you like to refine next?"
    return text


def _reply_asks_for_info(reply: str) -> bool:
    """True if the reply is asking a clarifying question—do not search in that case."""
    if not reply or not isinstance(reply, str):
        return False
    r = reply.strip()
    if r.endswith("?"):
        return True
    import re
    asking = (
        r"could you (please )?(tell|share)|what (is|are) (the )?(occasion|budget|recipient)",
        r"do you have (a )?(budget|occasion)|tell me (about |the )?(occasion|budget|interests)",
        r"please (share|tell|let me know)|(occasion|budget|interests) (for|of) (the )?gift",
        r"is it (a )?(birthday|anniversary|wedding|holiday)",
    )
    for pat in asking:
        if re.search(pat, r, re.I):
            return True
    return False


def _enforce_budget_on_query(query: str, gift_context: Optional[dict]) -> str:
    """Normalize budget phrase in a query from gift_context constraints."""
    if not isinstance(query, str):
        return query
    q = query.strip()
    if not q or not isinstance(gift_context, dict):
        return q
    bmin = gift_context.get("budget_min")
    bmax = gift_context.get("budget_max")
    try:
        has_min = isinstance(bmin, (int, float)) and float(bmin) > 0
        has_max = isinstance(bmax, (int, float)) and float(bmax) > 0
    except Exception:
        has_min = False
        has_max = False
    if not has_min and not has_max:
        return q

    import re
    # Strip existing budget phrases to avoid conflicts like "under $500 between $300 and $500"
    q = re.sub(r"\b(?:under|below|over|above)\s*\$?\s*\d+(?:[.,]\d+)?\b", "", q, flags=re.I)
    q = re.sub(r"\bbetween\s*\$?\s*\d+(?:[.,]\d+)?\s*(?:-|to|and)\s*\$?\s*\d+(?:[.,]\d+)?\b", "", q, flags=re.I)
    q = re.sub(r"\s{2,}", " ", q).strip()

    if has_min and has_max:
        budget_phrase = f"between ${int(float(bmin))} and ${int(float(bmax))}"
    elif has_max:
        # If only an upper limit is known (e.g., "up to $300"),
        # search a tighter range to improve relevance.
        max_v = int(float(bmax))
        min_v = max(1, int(round(max_v / 2)))
        budget_phrase = f"between ${min_v} and ${max_v}"
    else:
        budget_phrase = f"over ${int(float(bmin))}"
    return f"{q} {budget_phrase}".strip()


def _apply_budget_to_query_subtitles(query_subtitles: list, gift_context: Optional[dict]) -> list:
    if not isinstance(query_subtitles, list) or not query_subtitles:
        return query_subtitles
    out = []
    for q, s in query_subtitles:
        out.append((_enforce_budget_on_query(str(q or "").strip(), gift_context), s))
    return out


def _query_key(q: str) -> str:
    return " ".join(str(q or "").strip().lower().split())


def _normalize_products_by_query(items: Optional[list]) -> list:
    if not isinstance(items, list):
        return []
    out = []
    for row in items:
        if not isinstance(row, dict):
            continue
        q = str(row.get("query") or "").strip()
        products = row.get("products")
        if not q or not isinstance(products, list) or not products:
            continue
        subtitle = str(row.get("subtitle") or "").strip()
        clean_products = [p for p in products if isinstance(p, dict) and (p.get("title") or p.get("link"))]
        if not clean_products:
            continue
        out.append({"query": q, "subtitle": subtitle, "products": clean_products})
    return out


def _normalize_feedback_products(items: Optional[list]) -> list:
    if not isinstance(items, list):
        return []
    out = []
    seen = set()
    for row in items[:40]:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or "").strip()
        if not title:
            continue
        key = " ".join(title.lower().split())
        if not key or key in seen:
            continue
        seen.add(key)
        price = str(row.get("price") or "").strip()
        out.append({"title": title[:220], "price": price[:64]})
    return out


def _message_asks_about_shortlist(message: str) -> bool:
    m = str(message or "").strip().lower()
    if not m:
        return False
    triggers = (
        "which one", "which is best", "which one is best", "best one", "better one",
        "compare", "difference", "worth", "value", "reviews", "rating", "reliable",
        "should i buy", "recommend from", "among these", "between these", "between them",
        "pick one", "top one", "which should i pick", "which should i choose", "from these",
        "best option", "pick the best", "choose the best", "pick best option", "choose best option",
        "pick the best option", "choose the best option",
    )
    return any(t in m for t in triggers)


def _is_social_filler_message(message: str) -> bool:
    """Detect short gratitude/acknowledgement turns that should not trigger new search."""
    m = str(message or "").strip().lower()
    if not m:
        return False
    m = re.sub(r"[^\w\s]", " ", m)
    m = re.sub(r"\s{2,}", " ", m).strip()
    if not m:
        return False
    explicit_action_terms = (
        "search", "show", "more", "another", "different", "revise", "refine", "compare",
        "best", "which", "pick", "buy", "recommend", "options", "budget", "under", "between",
        "surprise", "trending", "explore",
    )
    if any(t in m for t in explicit_action_terms):
        return False
    filler_patterns = (
        r"^(thanks|thank you|thankyou|thx|ty)$",
        r"^(nice|great|awesome|perfect|cool|good)\s+(products|options|ideas|suggestions)$",
        r"^(nice|great|awesome|perfect|cool|good)\s+(products|options|ideas|suggestions)\s+(thanks|thank you|thx|ty)$",
        r"^(looks good|sounds good|that works|good stuff)$",
        r"^(appreciate it|much appreciated)$",
    )
    if any(re.search(p, m, re.I) for p in filler_patterns):
        return True
    token_count = len(m.split())
    if token_count <= 5 and ("thank" in m or "thanks" in m):
        return True
    return False


def _llm_requests_shortlist(parsed: Optional[dict], previous_products_by_query: list) -> bool:
    """LLM-directed shortlist routing (no keyword/rule matching on user text)."""
    return bool(
        previous_products_by_query
        and isinstance(parsed, dict)
        and parsed.get("shortlist_intent") is True
    )


def _answer_from_shortlist(
    message: str,
    history: list,
    gift_context: Optional[dict],
    previous_products_by_query: list,
) -> Optional[str]:
    shortlist = _normalize_products_by_query(previous_products_by_query)
    if not shortlist:
        return None
    payload = {
        "user_message": message,
        "gift_context": gift_context or {},
        "recent_history": history[-8:] if isinstance(history, list) else [],
        "shortlisted_products_by_query": shortlist,
    }
    qa_messages = [
        {"role": "developer", "content": SHORTLIST_QA_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    try:
        resp = _call_llm(qa_messages, stream=False)
        raw = (resp.choices[0].message.content or "").strip()
        parsed = _safe_json_loads(raw)
        reply = parsed.get("reply") if isinstance(parsed, dict) else None
        if isinstance(reply, str) and reply.strip():
            return _sanitize_reply_markdown_artifacts(reply.strip())
    except Exception as e:
        logger.warning("Shortlist QA call failed: %s", e)
    return None


def _is_open_explore_intent(message: str) -> bool:
    m = str(message or "").strip().lower()
    if not m:
        return False
    patterns = (
        r"\bshow\s+me\s+trending\b",
        r"\bhelp\s+me\s+explore\b",
        r"\blet'?s\s+explore\b",
        r"\bsurprise\s+me\b",
        r"\btrending\b",
        r"\bmore\s+ideas\b",
        r"\bopen\s+to\s+ideas\b",
    )
    return any(re.search(p, m, re.I) for p in patterns)


def _is_affirmation(message: str) -> bool:
    m = str(message or "").strip().lower()
    return m in {"yes", "y", "ok", "okay", "sure", "sounds good", "go ahead", "yep"}


def _allow_explore_without_interests(message: str, history: list) -> bool:
    # Only explicit user intent should unlock explore-without-interests.
    # Short affirmations like "yes/ok" should not bypass interest collection.
    return _is_open_explore_intent(message)


def _validate_payload(obj: dict) -> tuple[bool, str]:
    if not isinstance(obj, dict):
        return (False, "not an object")
    required = ("reply", "gift_context", "search_strategy")
    for k in required:
        if k not in obj:
            return (False, f"missing required key: {k}")
    if "reply" not in obj or not isinstance(obj.get("reply"), str) or not obj.get("reply", "").strip():
        return (False, "missing/invalid reply")
    if not isinstance(obj.get("gift_context"), dict):
        return (False, "missing/invalid gift_context")
    strategy = obj.get("search_strategy")
    if strategy is not None and not isinstance(strategy, dict):
        return (False, "missing/invalid search_strategy")
    return (True, "")


def _sse(event: str, data):
    try:
        payload = json.dumps(data, ensure_ascii=False)
    except Exception:
        payload = json.dumps({"ok": False})
    return f"event: {event}\ndata: {payload}\n\n"


def _resolve_search_queries(
    parsed: Optional[dict],
    raw_text: str,
    message: str,
    previous_queries: Optional[list],
    gift_context: Optional[dict],
    allow_explore_without_interests: bool = False,
) -> list:
    """Resolve search queries only from structured JSON fields (no text fallbacks)."""
    if not isinstance(parsed, dict):
        return []
    effective_ctx = merge_search_state(
        gift_context,
        parsed.get("gift_context") if isinstance(parsed.get("gift_context"), dict) else {}
    )
    strategy = parsed.get("search_strategy")
    if strategy is None:
        return []
    if not isinstance(strategy, dict):
        return []
    mode = str(strategy.get("mode") or "").strip().lower()
    # Hard gate: never search unless we have interests/hobbies or an explicit product.
    interests = effective_ctx.get("interests") if isinstance(effective_ctx, dict) else None
    has_interests = isinstance(interests, list) and any(str(i or "").strip() for i in interests)
    has_product = bool(str((effective_ctx or {}).get("product") or "").strip()) if isinstance(effective_ctx, dict) else False
    # Also require that interest/product didn't come only from an affirmation turn.
    # This prevents LLM-inferred products from bypassing discovery on "yes/ok".
    incoming_ctx = gift_context if isinstance(gift_context, dict) else {}
    incoming_interests = incoming_ctx.get("interests")
    incoming_has_interests = isinstance(incoming_interests, list) and any(str(i or "").strip() for i in incoming_interests)
    incoming_has_product = bool(str(incoming_ctx.get("product") or "").strip())
    if _is_affirmation(message) and not incoming_has_interests and not incoming_has_product and not _is_open_explore_intent(message):
        return []
    if not (has_interests or has_product):
        has_recipient = bool(str((effective_ctx or {}).get("recipient") or "").strip()) if isinstance(effective_ctx, dict) else False
        has_occasion = bool(str((effective_ctx or {}).get("occasion") or "").strip()) if isinstance(effective_ctx, dict) else False
        bmin = (effective_ctx or {}).get("budget_min") if isinstance(effective_ctx, dict) else None
        bmax = (effective_ctx or {}).get("budget_max") if isinstance(effective_ctx, dict) else None
        has_budget = bmin is not None or bmax is not None
        if not (allow_explore_without_interests and mode == "explore" and has_recipient and has_occasion and has_budget):
            return []
    query_subtitles = []
    for q in (strategy.get("queries") or [])[:3]:
        if isinstance(q, dict):
            query = (q.get("query") or "").strip()
            sub = (q.get("subtitle") or "").strip()
        else:
            query = str(q or "").strip()
            sub = ""
        if query:
            query_subtitles.append((query, sub))
    return _apply_budget_to_query_subtitles(query_subtitles, effective_ctx)


_init_abuse_db()


def gift_advisor_chat():
    """POST /gift_advisor - Chat endpoint for gift recommendations."""
    try:
        if request.method == "OPTIONS":
            return ("", 204)

        data = request.get_json(silent=True) or {}
        message = (data.get("message") or "").strip()
        if not message:
            return jsonify({"error": "Missing message"}), 400
        req_started = time.perf_counter()
        device_id = _normalize_device_id(data.get("device_id"))
        session_id = _normalize_session_id(data.get("session_id"), device_id)
        system_abuser_flag = _check_and_update_abuse(device_id, message)

        occasion = (data.get("occasion") or "").strip()
        budget_min = data.get("budget_min")
        budget_max = data.get("budget_max")
        history = _normalize_history(data.get("history") or [])
        gift_context = data.get("gift_context")
        liked_products = _normalize_feedback_products(data.get("liked_products"))
        disliked_products = _normalize_feedback_products(data.get("disliked_products"))
        people_profiles = _normalize_people_profiles(data.get("people_profiles"))
        active_profile_id = str(data.get("active_profile_id") or "").strip()
        incoming_active_profile_id = active_profile_id
        previous_queries = data.get("previous_queries")
        previous_products_by_query = _normalize_products_by_query(data.get("previous_products_by_query"))
        if not isinstance(gift_context, dict):
            gift_context = None
        if gift_context is None:
            gift_context = {}
        if liked_products:
            gift_context["liked_products"] = liked_products
        if disliked_products:
            gift_context["disliked_products"] = disliked_products
        # Resolve active person profile and context for this turn.
        effective_input_context, active_profile_id, people_profiles = _build_effective_input_context(
            people_profiles=people_profiles,
            active_profile_id=active_profile_id,
            gift_context=gift_context,
            message=message,
            occasion=occasion,
            budget_min=budget_min,
            budget_max=budget_max,
        )
        profile_switched = bool(
            incoming_active_profile_id
            and active_profile_id
            and str(incoming_active_profile_id) != str(active_profile_id)
        )
        if profile_switched:
            # Prevent old person's shortlist/query memory from leaking into this turn.
            previous_queries = []
            previous_products_by_query = []

        if len(history) <= 1:
            _track_telemetry_event(
                "session_started",
                device_id=device_id,
                session_id=session_id,
                props={
                    "history_len": len(history),
                    "stream_requested": bool(data.get("stream")),
                },
            )

        accept = (request.headers.get("Accept") or "").lower()
        wants_sse = "text/event-stream" in accept
        wants_stream = bool(data.get("stream")) or wants_sse

        if system_abuser_flag == 1:
            latency_ms = int((time.perf_counter() - req_started) * 1000)
            _track_telemetry_event(
                "chat_turn_completed",
                device_id=device_id,
                session_id=session_id,
                latency_ms=latency_ms,
                props={
                    "blocked": True,
                    "has_search": False,
                    "products_count": 0,
                    "profile_switched": bool(profile_switched),
                    "stream_mode": bool(wants_stream),
                },
            )
            payload = {
                "reply": GENERIC_ABUSE_REPLY,
                "gift_context": gift_context or {},
                "system_abuser_flag": 1,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            if not wants_stream:
                return jsonify(payload)

            @stream_with_context
            def blocked_gen():
                yield _sse("meta", {"ok": True, "stream": True, "ts": datetime.now(timezone.utc).isoformat()})
                yield _sse("final", payload)
                yield _sse("done", {"ok": True})

            headers = {
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
            return Response(blocked_gen(), headers=headers)

        context_bits = []
        if occasion:
            context_bits.append(f"occasion={occasion}")
        if active_profile_id:
            context_bits.append(f"active_profile_id={active_profile_id}")
        if effective_input_context:
            context_bits.append(f"gift_context={json.dumps(effective_input_context, ensure_ascii=False)}")
        if people_profiles.get("profiles"):
            profile_summary = []
            for p in people_profiles.get("profiles", [])[:8]:
                ctx = p.get("context") if isinstance(p.get("context"), dict) else {}
                profile_summary.append(
                    {
                        "id": p.get("id"),
                        "recipient": ctx.get("recipient"),
                        "occasion": ctx.get("occasion"),
                        "budget_min": ctx.get("budget_min"),
                        "budget_max": ctx.get("budget_max"),
                        "interests": ctx.get("interests"),
                        "liked_products": [x.get("title") for x in (ctx.get("liked_products") or []) if isinstance(x, dict)][:5],
                        "disliked_products": [x.get("title") for x in (ctx.get("disliked_products") or []) if isinstance(x, dict)][:5],
                    }
                )
            context_bits.append(f"people_profiles={json.dumps(profile_summary, ensure_ascii=False)}")
        if previous_products_by_query:
            prev_q = [r.get("query") for r in previous_products_by_query if r.get("query")]
            context_bits.append(f"previous_search_queries={json.dumps(prev_q, ensure_ascii=False)}")
            context_bits.append("previous_products_shown=true")
        elif isinstance(previous_queries, list) and previous_queries:
            context_bits.append(f"previous_search_queries={json.dumps(previous_queries)}")
            context_bits.append("previous_products_shown=true")
        elif history:
            context_bits.append("(no previous_search_queries—derive product, recipient, budget from conversation history below)")

        context_msg = "GIFT_CONTEXT: " + (", ".join(context_bits) if context_bits else "none")
        allow_explore_no_interests = _allow_explore_without_interests(message, history)

        input_messages = [
            {"role": "developer", "content": GIFT_ADVISOR_SYSTEM_PROMPT},
            {"role": "developer", "content": context_msg},
            {"role": "developer", "content": GIFT_ADVISOR_ENRICHMENT},
        ]
        input_messages.extend(history)
        if not (history and history[-1]["role"] == "user" and history[-1]["content"] == message):
            input_messages.append({"role": "user", "content": message})

        def _parse_or_fix_json(raw_text: str):
            raw_text = (raw_text or "").strip()
            parsed = _safe_json_loads(raw_text) if raw_text else None
            ok, _ = _validate_payload(parsed) if isinstance(parsed, dict) else (False, "parse failed")
            if ok:
                return parsed, raw_text
            return None, raw_text

        def _finalize_payload(parsed: Optional[dict], raw_fallback: str, products_by_query: Optional[list] = None):
            reply = None
            out_gift_context = None
            if isinstance(parsed, dict):
                reply = parsed.get("reply")
                if isinstance(parsed.get("gift_context"), dict):
                    out_gift_context = parsed.get("gift_context")
            if not isinstance(reply, str) or not reply.strip():
                reply = raw_fallback.strip() if raw_fallback else "[No content returned]"
            reply = _strip_search_queries_from_reply(_compact_ui_text(reply))
            merged_out_context = merge_search_state(effective_input_context, out_gift_context or {})
            reply = _avoid_reasking_known_fields(reply, merged_out_context)
            reply = _ensure_next_step(reply, merged_out_context, has_products=bool(products_by_query))
            updated_profiles = _update_active_profile_context(people_profiles, active_profile_id, merged_out_context)
            payload = {
                "reply": reply,
                "gift_context": merged_out_context,
                "people_profiles": updated_profiles,
                "active_profile_id": active_profile_id,
                "system_abuser_flag": int(system_abuser_flag),
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            if products_by_query:
                payload["products_by_query"] = products_by_query
            return payload


        if not wants_stream:
            resp = _call_llm(input_messages, stream=False)
            raw = (resp.choices[0].message.content or "").strip()
            parsed, _ = _parse_or_fix_json(raw)
            effective_gift_context = merge_search_state(
                effective_input_context,
                parsed.get("gift_context") if isinstance(parsed, dict) and isinstance(parsed.get("gift_context"), dict) else {}
            )
            if _is_social_filler_message(message):
                if not isinstance(parsed, dict):
                    reply_fallback = _strip_search_queries_from_reply(_compact_ui_text(raw)) or "Thanks! Happy to help."
                    parsed = {"reply": reply_fallback, "gift_context": {}, "search_strategy": None}
                parsed["search_strategy"] = None
            # If user asks about already shown products, do shortlist analysis only (no new search).
            if previous_products_by_query and _message_asks_about_shortlist(message):
                shortlist_reply = _answer_from_shortlist(
                    message=message,
                    history=history,
                    gift_context=effective_gift_context,
                    previous_products_by_query=previous_products_by_query,
                )
                if shortlist_reply:
                    if not isinstance(parsed, dict):
                        parsed = {"reply": shortlist_reply, "gift_context": {}, "search_strategy": None}
                    parsed["reply"] = shortlist_reply
                    parsed["search_strategy"] = None

            products_by_query = []
            query_subtitles = _resolve_search_queries(
                parsed, raw, message, previous_queries, effective_input_context, allow_explore_without_interests=allow_explore_no_interests
            )
            if query_subtitles:
                prev_map = {_query_key(r.get("query")): r for r in previous_products_by_query}
                new_query_subtitles = [(q, s) for q, s in query_subtitles if _query_key(q) not in prev_map]
                unchanged = len(new_query_subtitles) == 0
                queries = [q for q, _ in query_subtitles]
                logger.info(
                    "[GIFT_ADVISOR] product_search | message=%r | history=%s | previous_queries=%s | gift_context=%s | context=%s | llm_search_queries=%s | resolved=%s | queries_submitted=%s",
                    message,
                    json.dumps(history[-6:] if history else [], ensure_ascii=False),
                    previous_queries,
                    gift_context,
                    context_msg[:200] + "..." if len(context_msg) > 200 else context_msg,
                    parsed.get("search_strategy") if isinstance(parsed, dict) else None,
                    query_subtitles,
                    [q for q, _ in new_query_subtitles] if not unchanged else [],
                )
                if unchanged:
                    # Same query set as already shown: avoid repetitive re-render/search.
                    if previous_products_by_query and _message_asks_about_shortlist(message):
                        shortlist_reply = _answer_from_shortlist(
                            message=message,
                            history=history,
                            gift_context=effective_gift_context,
                            previous_products_by_query=previous_products_by_query,
                        )
                        if shortlist_reply and isinstance(parsed, dict):
                            parsed["reply"] = shortlist_reply
                            parsed["search_strategy"] = None
                else:
                    raw_results = scrape_amazon_searches(
                        [q for q, _ in new_query_subtitles], products_per_search=PRODUCTS_PER_SEARCH
                    )
                    products_by_query = _rank_products_by_sections(
                        raw_results=raw_results,
                        query_subtitles=new_query_subtitles,
                        message=message,
                        history=history,
                        gift_context=effective_gift_context,
                    )
                    _track_telemetry_event(
                        "search_executed",
                        device_id=device_id,
                        session_id=session_id,
                        props={
                            "queries_count": len(new_query_subtitles),
                            "queries": [q for q, _ in new_query_subtitles][:3],
                            "products_count": sum(
                                len(r.get("products") or [])
                                for r in (products_by_query or [])
                                if isinstance(r, dict)
                            ),
                            "stream_mode": False,
                        },
                    )
            else:
                logger.info(
                    "[GIFT_ADVISOR] no_product_search | message=%r | history=%s | previous_queries=%s | gift_context=%s | llm_search_queries=%s | resolved=[]",
                    message,
                    json.dumps(history[-6:] if history else [], ensure_ascii=False),
                    previous_queries,
                    gift_context,
                    parsed.get("search_strategy") if isinstance(parsed, dict) else None,
                )
            payload = _finalize_payload(parsed, raw, products_by_query=products_by_query if products_by_query else None)
            latency_ms = int((time.perf_counter() - req_started) * 1000)
            _track_telemetry_event(
                "chat_turn_completed",
                device_id=device_id,
                session_id=session_id,
                latency_ms=latency_ms,
                props={
                    "blocked": False,
                    "has_search": bool(query_subtitles),
                    "queries_count": len(query_subtitles or []),
                    "products_count": sum(
                        len(r.get("products") or [])
                        for r in (products_by_query or [])
                        if isinstance(r, dict)
                    ),
                    "profile_switched": bool(profile_switched),
                    "shortlist_intent": bool(_llm_requests_shortlist(parsed, previous_products_by_query)),
                    "stream_mode": False,
                },
            )
            return jsonify(payload)

        @stream_with_context
        def gen():
            yield _sse("meta", {"ok": True, "stream": True, "ts": datetime.now(timezone.utc).isoformat()})
            raw_buf = ""
            try:
                stream = _call_llm(input_messages, stream=True)
                emitted_len = 0
                in_reply_field = False
                esc = False

                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content
                    if not (isinstance(delta, str) and delta):
                        continue
                    raw_buf += delta
                    # Re-parse full buffer each chunk to extract reply (may span chunks)
                    reply_buf = ""
                    s = raw_buf
                    i = 0
                    while i < len(s):
                        if not in_reply_field:
                            idx = s.find('"reply"', i)
                            if idx < 0:
                                break
                            j = idx + len('"reply"')
                            while j < len(s) and s[j] in " \t\r\n":
                                j += 1
                            if j >= len(s) or s[j] != ":":
                                i = j
                                continue
                            j += 1
                            while j < len(s) and s[j] in " \t\r\n":
                                j += 1
                            if j >= len(s) or s[j] != '"':
                                i = j
                                continue
                            in_reply_field = True
                            esc = False
                            i = j + 1
                            continue
                        ch = s[i]
                        if esc:
                            if ch == "n":
                                reply_buf += "\n"
                            elif ch == "t":
                                reply_buf += "\t"
                            elif ch == "r":
                                reply_buf += "\r"
                            elif ch == '"':
                                reply_buf += '"'
                            elif ch == "\\":
                                reply_buf += "\\"
                            elif ch == "u" and i + 4 < len(s):
                                try:
                                    reply_buf += chr(int(s[i + 1 : i + 5], 16))
                                    i += 4
                                except Exception:
                                    reply_buf += "\\u"
                            else:
                                reply_buf += ch
                            esc = False
                        else:
                            if ch == "\\":
                                esc = True
                            elif ch == '"':
                                in_reply_field = False
                            else:
                                reply_buf += ch
                        i += 1

                    if len(reply_buf) > emitted_len:
                        out_delta = reply_buf[emitted_len:]
                        emitted_len = len(reply_buf)
                        if out_delta:
                            yield _sse("delta", {"text": out_delta})

                parsed, _ = _parse_or_fix_json(raw_buf)
                effective_gift_context = merge_search_state(
                    effective_input_context,
                    parsed.get("gift_context") if isinstance(parsed, dict) and isinstance(parsed.get("gift_context"), dict) else {}
                )
                if _is_social_filler_message(message):
                    if not isinstance(parsed, dict):
                        reply_fallback = _strip_search_queries_from_reply(_compact_ui_text(raw_buf)) or "Thanks! Happy to help."
                        parsed = {"reply": reply_fallback, "gift_context": {}, "search_strategy": None}
                    parsed["search_strategy"] = None
                # If user asks about already shown products, do shortlist analysis only (no new search).
                if previous_products_by_query and _message_asks_about_shortlist(message):
                    shortlist_reply = _answer_from_shortlist(
                        message=message,
                        history=history,
                        gift_context=effective_gift_context,
                        previous_products_by_query=previous_products_by_query,
                    )
                    if shortlist_reply:
                        if not isinstance(parsed, dict):
                            parsed = {"reply": shortlist_reply, "gift_context": {}, "search_strategy": None}
                        parsed["reply"] = shortlist_reply
                        parsed["search_strategy"] = None

                products_by_query = []
                query_subtitles = _resolve_search_queries(
                    parsed, raw_buf, message, previous_queries, effective_input_context, allow_explore_without_interests=allow_explore_no_interests
                )
                if query_subtitles:
                    prev_map = {_query_key(r.get("query")): r for r in previous_products_by_query}
                    new_query_subtitles = [(q, s) for q, s in query_subtitles if _query_key(q) not in prev_map]
                    unchanged = len(new_query_subtitles) == 0
                    queries = [q for q, _ in query_subtitles]
                    logger.info(
                        "[GIFT_ADVISOR] product_search (stream) | message=%r | history=%s | previous_queries=%s | gift_context=%s | context=%s | llm_search_queries=%s | resolved=%s | queries_submitted=%s",
                        message,
                        json.dumps(history[-6:] if history else [], ensure_ascii=False),
                        previous_queries,
                        gift_context,
                        context_msg[:200] + "..." if len(context_msg) > 200 else context_msg,
                        parsed.get("search_strategy") if isinstance(parsed, dict) else None,
                        query_subtitles,
                        [q for q, _ in new_query_subtitles] if not unchanged else [],
                    )
                    if unchanged:
                        # Same query set as already shown: avoid repetitive re-render/search.
                        if previous_products_by_query and _message_asks_about_shortlist(message):
                            shortlist_reply = _answer_from_shortlist(
                                message=message,
                                history=history,
                                gift_context=effective_gift_context,
                                previous_products_by_query=previous_products_by_query,
                            )
                            if shortlist_reply and isinstance(parsed, dict):
                                parsed["reply"] = shortlist_reply
                                parsed["search_strategy"] = None
                    else:
                        incremental_queries = [q for q, _ in new_query_subtitles]
                        yield _sse("products_loading", {"queries": incremental_queries})
                        raw_results = scrape_amazon_searches(
                            incremental_queries, products_per_search=PRODUCTS_PER_SEARCH
                        )
                        products_by_query = _rank_products_by_sections(
                            raw_results=raw_results,
                            query_subtitles=new_query_subtitles,
                            message=message,
                            history=history,
                            gift_context=effective_gift_context,
                        )
                        _track_telemetry_event(
                            "search_executed",
                            device_id=device_id,
                            session_id=session_id,
                            props={
                                "queries_count": len(incremental_queries),
                                "queries": incremental_queries[:3],
                                "products_count": sum(
                                    len(r.get("products") or [])
                                    for r in (products_by_query or [])
                                    if isinstance(r, dict)
                                ),
                                "stream_mode": True,
                            },
                        )
                else:
                    logger.info(
                        "[GIFT_ADVISOR] no_product_search (stream) | message=%r | previous_queries=%s | gift_context=%s | llm_search_queries=%s",
                        message,
                        previous_queries,
                        gift_context,
                        parsed.get("search_strategy") if isinstance(parsed, dict) else None,
                    )
                payload = _finalize_payload(parsed, raw_buf, products_by_query=products_by_query if products_by_query else None)
                latency_ms = int((time.perf_counter() - req_started) * 1000)
                _track_telemetry_event(
                    "chat_turn_completed",
                    device_id=device_id,
                    session_id=session_id,
                    latency_ms=latency_ms,
                    props={
                        "blocked": False,
                        "has_search": bool(query_subtitles),
                        "queries_count": len(query_subtitles or []),
                        "products_count": sum(
                            len(r.get("products") or [])
                            for r in (products_by_query or [])
                            if isinstance(r, dict)
                        ),
                        "profile_switched": bool(profile_switched),
                        "shortlist_intent": bool(_llm_requests_shortlist(parsed, previous_products_by_query)),
                        "stream_mode": True,
                    },
                )
                yield _sse("final", payload)
                yield _sse("done", {"ok": True})
            except Exception as e:
                logger.exception("Exception in gift_advisor stream")
                latency_ms = int((time.perf_counter() - req_started) * 1000)
                _track_telemetry_event(
                    "error_occurred",
                    device_id=device_id,
                    session_id=session_id,
                    latency_ms=latency_ms,
                    props={
                        "stage": "stream",
                        "error_type": type(e).__name__,
                        "stream_mode": True,
                    },
                )
                fallback_payload = {
                    "reply": "I hit a temporary issue while fetching product options. Please try again, and I will retry right away.",
                    "gift_context": effective_input_context if isinstance(effective_input_context, dict) else {},
                    "people_profiles": people_profiles,
                    "active_profile_id": active_profile_id,
                    "system_abuser_flag": int(system_abuser_flag),
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                yield _sse("final", fallback_payload)
                yield _sse("done", {"ok": False})

        headers = {
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return Response(gen(), headers=headers)

    except Exception as e:
        err_msg = str(e)
        try:
            req_data = request.get_json(silent=True) or {}
            err_device = _normalize_device_id(req_data.get("device_id"))
            err_session = _normalize_session_id(req_data.get("session_id"), err_device)
            _track_telemetry_event(
                "error_occurred",
                device_id=err_device,
                session_id=err_session,
                props={"stage": "route", "error_type": type(e).__name__, "stream_mode": False},
            )
        except Exception:
            pass
        print("Exception in gift_advisor_chat:", repr(e))
        return jsonify({"error": err_msg or "Server error"}), 500
