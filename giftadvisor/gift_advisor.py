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

logger = logging.getLogger("gift_advisor")

from product_search import scrape_amazon_searches

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
try:
    PRODUCTS_PER_SEARCH = max(10, min(int(os.getenv("PRODUCTS_PER_SEARCH", "20")), 30))
except Exception:
    PRODUCTS_PER_SEARCH = 20

ABUSE_DB_PATH = os.path.join(os.path.dirname(__file__), "logs", "abuse_flags.sqlite3")
GENERIC_ABUSE_REPLY = "I can help with gift suggestions, but I cannot process this request right now. Please try again with a normal, concise gift-related question."

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
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
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
    "interests": ["string"]
  },
  "search_strategy": {
    "mode": "specific | explore",
    "queries": [
      {
        "query": "string (Amazon search string)",
        "subtitle": "string (user-friendly display title, e.g. 'Thoughtful picks for her' or 'Gifts for her 45th'—NOT the raw query)"
      }
    ]
  } | null
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
9. Only return search_strategy (and trigger a search) when you have occasion + budget + (interests or product). Until then, ask for what's missing and return null. Do NOT search and ask in the same turn.
10. When the user has provided enough (occasion, budget, interests/product), return search_strategy with queries. Do NOT return null when you have enough context.
11. Treat values already present in gift_context (including UI-selected occasion/budget) as known facts; do not ask for them again.
12. Mention on-screen occasion/budget options ONLY when either occasion or budget is still missing. If occasion and budget are already known in gift_context, do NOT mention on-screen options.
13. Ask for recipient age (or age range) as part of exploration. If the recipient is a child/teen (e.g., son, daughter, kid, teenager, 13-17), collect age before any product search.
""".strip()

PRODUCT_RANKER_PROMPT = """
You are a product ranking assistant for gift recommendations.

Task:
- Select the best top_k products from candidate products for the given context.
- Prioritize: recipient fit, occasion fit, budget fit, interest/brand fit, and reliability signals.
- Reliability signals to prefer: higher rating, higher reviews, and stronger bought_last_month.
- Avoid repetition: do NOT select near-duplicate products that are very similar in title/description/specs (e.g., same model with only minor variant differences) unless unique options are unavailable.
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
            key = (title.lower() or link.lower()).strip()
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
        key = (c.get("title") or "").strip().lower()
        if key and key in used:
            continue
        if key:
            used.add(key)
        chosen_candidates.append(c)
        if len(chosen_candidates) >= top_k:
            break

    if len(chosen_candidates) < top_k:
        for c in candidates:
            key = (c.get("title") or "").strip().lower()
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
        used_titles = {(c.get("title") or "").strip().lower() for c in chosen_candidates if (c.get("title") or "").strip()}
        missing_queries = [q for q in pool_queries if q not in picked_queries]
        replacement = None
        for mq in missing_queries:
            for c in candidates:
                if str(c.get("query") or "").strip() != mq:
                    continue
                t = (c.get("title") or "").strip().lower()
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
    sections = []
    global_candidates = _flatten_product_candidates(raw_results, max_candidates=1000)

    for idx, q in enumerate(ordered_queries, start=1):
        row = next((r for r in raw_results if str(r.get("query") or "").strip() == q), None)
        row_candidates = _flatten_product_candidates([row], max_candidates=1000) if isinstance(row, dict) else []

        # Rank with full pool visibility, but keep section intent via query_hints.
        # Start with row-specific candidates and append remaining global ones.
        candidates = []
        seen = set()
        for c in row_candidates + global_candidates:
            cid = str(c.get("id") or "")
            title = (c.get("title") or "").strip().lower()
            key = cid or title
            if not key or key in seen:
                continue
            seen.add(key)
            candidates.append(c)

        if not candidates:
            continue
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
            t = (rc.get("title") or "").strip().lower()
            l = ((rc.get("product") or {}).get("link") or "").strip().lower()
            if t:
                row_keys.add(("t", t))
            if l:
                row_keys.add(("l", l))
        strict_ranked = []
        used_keys = set()
        for p in ranked or []:
            t = (p.get("title") or "").strip().lower()
            l = (p.get("link") or "").strip().lower()
            in_row = (("t", t) in row_keys) or (("l", l) in row_keys)
            if not in_row:
                continue
            key = ("t", t) if t else ("l", l)
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
                t = (p.get("title") or "").strip().lower()
                l = (p.get("link") or "").strip().lower()
                key = ("t", t) if t else ("l", l)
                if key in used_keys:
                    continue
                used_keys.add(key)
                strict_ranked.append(p)
                if len(strict_ranked) >= top_k:
                    break

        if not strict_ranked:
            continue
        sections.append({
            "query": f"category_{idx}",
            "subtitle": _friendly_section_title(q, subtitle_by_query.get(q, ""), idx),
            "products": strict_ranked[:top_k],
        })

    return sections


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
    text = (reply or "").strip()
    if not text:
        return text

    has_budget = gift_context.get("budget_min") is not None or gift_context.get("budget_max") is not None

    # If occasion is already known, strip only the occasion-ask fragments.
    if gift_context.get("occasion"):
        patterns = [
            r"(?i)\b(?:could\s+you(?:\s+please)?|can\s+you|please)\s+(?:tell|share)(?:\s+me)?(?:\s+what(?:'s|\s+is))?\s+(?:the\s+)?occasion[^?.!]*[?.!]?",
            r"(?i)\bwhat(?:'s|\s+is)\s+(?:the\s+)?occasion[^?.!]*[?.!]?",
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

    # Cleanup connective artifacts after fragment removal.
    text = re.sub(r"(?i)\b(and|also)\s*(?:,)?\s*(and|also)\b", r"\1", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,;.-")

    # Guard against awkward tiny remnants like "Great!"
    if len(text) < 12:
        has_interests = bool(gift_context.get("interests"))
        has_product = bool(str(gift_context.get("product") or "").strip())
        has_recipient = bool(str(gift_context.get("recipient") or "").strip())
        if not has_recipient:
            text = "Great! Who is the gift for?"
        elif not has_interests and not has_product:
            text = "Great! What is she into these days (hobbies, style, or favorite things)?"
        elif not has_budget:
            text = "Great! What budget range should I target?"
        else:
            text = "Great! Any specific style, brand, or type of gift you want to prioritize?"

    return _compact_ui_text(text)


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
        budget_phrase = f"under ${int(float(bmax))}"
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


def _message_asks_about_shortlist(message: str) -> bool:
    m = str(message or "").strip().lower()
    if not m:
        return False
    triggers = (
        "which", "best", "better", "compare", "difference", "worth", "value",
        "reviews", "rating", "reliable", "should i buy", "recommend from",
        "among these", "between these", "pick one", "top one", "why",
    )
    return ("?" in m) or any(t in m for t in triggers)


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
            return reply.strip()
    except Exception as e:
        logger.warning("Shortlist QA call failed: %s", e)
    return None


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
        device_id = _normalize_device_id(data.get("device_id"))
        system_abuser_flag = _check_and_update_abuse(device_id, message)

        occasion = (data.get("occasion") or "").strip()
        budget_min = data.get("budget_min")
        budget_max = data.get("budget_max")
        history = _normalize_history(data.get("history") or [])
        gift_context = data.get("gift_context")
        previous_queries = data.get("previous_queries")
        previous_products_by_query = _normalize_products_by_query(data.get("previous_products_by_query"))
        if not isinstance(gift_context, dict):
            gift_context = None
        # Promote selected occasion/budget chips into effective context so the LLM won't ask again.
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
        effective_input_context = merge_search_state(gift_context, ui_ctx)

        accept = (request.headers.get("Accept") or "").lower()
        wants_sse = "text/event-stream" in accept
        wants_stream = bool(data.get("stream")) or wants_sse

        if system_abuser_flag == 1:
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
        if effective_input_context:
            context_bits.append(f"gift_context={json.dumps(effective_input_context, ensure_ascii=False)}")
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
            payload = {
                "reply": reply,
                "gift_context": merged_out_context,
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
            products_by_query = []
            query_subtitles = _resolve_search_queries(parsed, raw, message, previous_queries, effective_input_context)
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
                products_by_query = []
                query_subtitles = _resolve_search_queries(
                    parsed, raw_buf, message, previous_queries, effective_input_context
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
                else:
                    logger.info(
                        "[GIFT_ADVISOR] no_product_search (stream) | message=%r | previous_queries=%s | gift_context=%s | llm_search_queries=%s",
                        message,
                        previous_queries,
                        gift_context,
                        parsed.get("search_strategy") if isinstance(parsed, dict) else None,
                    )
                payload = _finalize_payload(parsed, raw_buf, products_by_query=products_by_query if products_by_query else None)
                yield _sse("final", payload)
                yield _sse("done", {"ok": True})
            except Exception as e:
                yield _sse("error", {"ok": False, "error": "Server error"})
                print("Exception in gift_advisor stream:", repr(e))

        headers = {
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return Response(gen(), headers=headers)

    except Exception as e:
        err_msg = str(e)
        print("Exception in gift_advisor_chat:", repr(e))
        return jsonify({"error": err_msg or "Server error"}), 500
