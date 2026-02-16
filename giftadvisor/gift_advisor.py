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

logger = logging.getLogger("gift_advisor")

from product_search import scrape_amazon_searches

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

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
        temperature=0.6,
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
- Learn about the recipient: age range, interests, relationship to giver, personality
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
""".strip()

PRODUCT_RANKER_PROMPT = """
You are a product ranking assistant for gift recommendations.

Task:
- Select the best 3 products from candidate products for the given context.
- Prioritize: recipient fit, occasion fit, budget fit, interest/brand fit, product quality signals (rating/reviews).
- Reject obvious mismatches (e.g., infant/kids products when recipient is an adult wife/woman) unless explicitly requested.
- Do not invent products.
- Return ONLY valid JSON:
{
  "selected_ids": ["p1", "p4", "p2"],
  "reasons": ["short reason 1", "short reason 2", "short reason 3"]
}
""".strip()




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


def _flatten_product_candidates(raw_results: list, max_candidates: int = 15) -> list:
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
    top_k: int = 3,
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
    chosen = []
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
        chosen.append(c.get("product") or {})
        if len(chosen) >= top_k:
            break

    if len(chosen) < top_k:
        for c in candidates:
            key = (c.get("title") or "").strip().lower()
            if key and key in used:
                continue
            if key:
                used.add(key)
            chosen.append(c.get("product") or {})
            if len(chosen) >= top_k:
                break

    return chosen[:top_k]


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


def _validate_payload(obj: dict) -> tuple[bool, str]:
    if not isinstance(obj, dict):
        return (False, "not an object")
    if "reply" not in obj or not isinstance(obj.get("reply"), str) or not obj.get("reply", "").strip():
        return (False, "missing/invalid reply")
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
    """Resolve search queries. Trust LLM when it returns null—no search. Only use fallbacks when field is omitted."""
    query_subtitles = []
    if isinstance(parsed, dict):
        strategy = parsed.get("search_strategy")
        sq_list = parsed.get("search_queries")
        if strategy is None and "search_strategy" in parsed:
            return []
        reply_text = (parsed.get("reply") or "").strip()
        if _reply_asks_for_info(reply_text):
            return []
        if isinstance(strategy, dict):
            queries = strategy.get("queries") or []
            for q in queries[:3]:
                if isinstance(q, dict):
                    query = (q.get("query") or "").strip()
                    sub = (q.get("subtitle") or "").strip()
                else:
                    query = str(q or "").strip()
                    sub = ""
                if query:
                    query_subtitles.append((query, sub))
            return query_subtitles
        if "search_queries" in parsed:
            if isinstance(sq_list, list) and sq_list:
                query_subtitles = _parse_search_queries(sq_list)
            elif isinstance(parsed.get("search_query"), str):
                sq = (parsed.get("search_query") or "").strip()
                if sq:
                    query_subtitles = [(sq, "")]
            else:
                return []
        else:
            reply_text = (parsed.get("reply") or "").strip()
            if _reply_asks_for_info(reply_text):
                return []
            merged_ctx = merge_search_state(gift_context, parsed.get("gift_context") if isinstance(parsed.get("gift_context"), dict) else {})
            queries = []
            if merged_ctx and merged_ctx.get("product"):
                queries = _derive_queries_from_gift_context(merged_ctx, message)
            if not queries:
                queries = _extract_search_queries_from_reply((parsed.get("reply") or "").strip() or (raw_text or "").strip())
            if not queries:
                queries = _user_message_to_search_queries(message)
            if not queries and isinstance(previous_queries, list) and previous_queries:
                queries = [str(q).strip() for q in previous_queries[:3] if str(q).strip()]
            if not queries and merged_ctx:
                queries = _derive_queries_from_gift_context(merged_ctx, message)
            query_subtitles = [(q, "") for q in queries if q]
    else:
        reply_text = (raw_text or "").strip()
        parsed_reply = _extract_first_json_object(reply_text)
        if isinstance(parsed_reply, dict) and parsed_reply.get("reply"):
            reply_text = (parsed_reply.get("reply") or "").strip()
        if _reply_asks_for_info(reply_text):
            return []
        merged_ctx = gift_context
        queries = []
        if merged_ctx and merged_ctx.get("product"):
            queries = _derive_queries_from_gift_context(merged_ctx, message)
        if not queries:
            queries = _extract_search_queries_from_reply(reply_text)
        if not queries:
            queries = _user_message_to_search_queries(message)
        if not queries and isinstance(previous_queries, list) and previous_queries:
            queries = [str(q).strip() for q in previous_queries[:3] if str(q).strip()]
        if not queries and merged_ctx:
            queries = _derive_queries_from_gift_context(merged_ctx, message)
        query_subtitles = [(q, "") for q in queries if q]
    return query_subtitles



def gift_advisor_chat():
    """POST /gift_advisor - Chat endpoint for gift recommendations."""
    try:
        if request.method == "OPTIONS":
            return ("", 204)

        data = request.get_json(silent=True) or {}
        message = (data.get("message") or "").strip()
        if not message:
            return jsonify({"error": "Missing message"}), 400

        occasion = (data.get("occasion") or "").strip()
        history = _normalize_history(data.get("history") or [])
        gift_context = data.get("gift_context")
        previous_queries = data.get("previous_queries")
        if not isinstance(gift_context, dict):
            gift_context = None

        accept = (request.headers.get("Accept") or "").lower()
        wants_sse = "text/event-stream" in accept
        wants_stream = bool(data.get("stream")) or wants_sse

        context_bits = []
        if occasion:
            context_bits.append(f"occasion={occasion}")
        if gift_context:
            context_bits.append(f"gift_context={json.dumps(gift_context, ensure_ascii=False)}")
        if isinstance(previous_queries, list) and previous_queries:
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
            parsed = _extract_first_json_object(raw_text) if raw_text else None
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
            payload = {
                "reply": reply,
                "gift_context": out_gift_context or gift_context,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            if products_by_query:
                payload["products_by_query"] = products_by_query
            return payload


        if not wants_stream:
            resp = _call_llm(input_messages, stream=False)
            raw = (resp.choices[0].message.content or "").strip()
            parsed, _ = _parse_or_fix_json(raw)
            products_by_query = []
            query_subtitles = _resolve_search_queries(parsed, raw, message, previous_queries, gift_context)
            if query_subtitles:
                queries = [q for q, _ in query_subtitles]
                logger.info(
                    "[GIFT_ADVISOR] product_search | message=%r | history=%s | previous_queries=%s | gift_context=%s | context=%s | llm_search_queries=%s | resolved=%s | queries_submitted=%s",
                    message,
                    json.dumps(history[-6:] if history else [], ensure_ascii=False),
                    previous_queries,
                    gift_context,
                    context_msg[:200] + "..." if len(context_msg) > 200 else context_msg,
                    parsed.get("search_queries") if isinstance(parsed, dict) else None,
                    query_subtitles,
                    queries,
                )
                raw_results = scrape_amazon_searches(queries, products_per_search=5)
                candidates = _flatten_product_candidates(raw_results, max_candidates=15)
                top3 = _rank_products_with_llm(
                    candidates=candidates,
                    message=message,
                    history=history,
                    gift_context=gift_context,
                    query_subtitles=query_subtitles,
                    top_k=3,
                )
                products_by_query = [{"query": "Top picks", "subtitle": "Top picks for you", "products": top3}] if top3 else []
            else:
                logger.info(
                    "[GIFT_ADVISOR] no_product_search | message=%r | history=%s | previous_queries=%s | gift_context=%s | llm_search_queries=%s | resolved=[]",
                    message,
                    json.dumps(history[-6:] if history else [], ensure_ascii=False),
                    previous_queries,
                    gift_context,
                    parsed.get("search_queries") if isinstance(parsed, dict) else None,
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
                products_by_query = []
                query_subtitles = _resolve_search_queries(
                    parsed, raw_buf, message, previous_queries, gift_context
                )
                if query_subtitles:
                    queries = [q for q, _ in query_subtitles]
                    logger.info(
                        "[GIFT_ADVISOR] product_search (stream) | message=%r | history=%s | previous_queries=%s | gift_context=%s | context=%s | llm_search_queries=%s | resolved=%s | queries_submitted=%s",
                        message,
                        json.dumps(history[-6:] if history else [], ensure_ascii=False),
                        previous_queries,
                        gift_context,
                        context_msg[:200] + "..." if len(context_msg) > 200 else context_msg,
                        parsed.get("search_queries") if isinstance(parsed, dict) else None,
                        query_subtitles,
                        queries,
                    )
                    yield _sse("products_loading", {"queries": queries})
                    raw_results = scrape_amazon_searches(queries, products_per_search=5)
                    candidates = _flatten_product_candidates(raw_results, max_candidates=15)
                    top3 = _rank_products_with_llm(
                        candidates=candidates,
                        message=message,
                        history=history,
                        gift_context=gift_context,
                        query_subtitles=query_subtitles,
                        top_k=3,
                    )
                    products_by_query = [{"query": "Top picks", "subtitle": "Top picks for you", "products": top3}] if top3 else []
                else:
                    logger.info(
                        "[GIFT_ADVISOR] no_product_search (stream) | message=%r | previous_queries=%s | gift_context=%s | llm_search_queries=%s",
                        message,
                        previous_queries,
                        gift_context,
                        parsed.get("search_queries") if isinstance(parsed, dict) else None,
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
