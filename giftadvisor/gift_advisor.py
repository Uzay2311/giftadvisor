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

GIFT_ADVISOR_SYSTEM_PROMPT = """
You are a warm, knowledgeable Gift Advisor. Your goal is to help people find the perfect gift for a specific occasion.

You are NOT a general chat companion. You focus exclusively on gift-giving advice.

Core behaviors:
- Ask about the occasion (birthday, anniversary, wedding, holiday, graduation, etc.)
- Learn about the recipient: age range, interests, relationship to giver, personality
- Understand budget constraints and preferences
- Suggest thoughtful, personalized gift ideas with brief reasoning
- Keep responses concise and actionable (2-4 sentences typically)
- Use bullet points for multiple suggestions
- Be encouraging and creative—help people feel confident in their choice

Formatting:
- Use \\n for new lines. Avoid \\n\\n.
- Prefer bullets for lists.
- Keep suggestions skimmable and scannable.
""".strip()

GIFT_ADVISOR_ENRICHMENT = """
Return structured JSON only:
{
  "reply": "string (your response to the user, UI-ready, use \\n not \\n\\n)",
  "gift_context": {
    "occasion": "string | null",
    "recipient_info": "string | null",
    "budget": "string | null",
    "interests": ["string"],
    "suggestions_so_far": ["string"]
  } | null,
  "search_queries": [{"query": "string", "subtitle": "string"} | "string"] | null
}

Rules:
- gift_context: update as the conversation progresses. ALWAYS set recipient_info (e.g. "wife", "mom") and budget when user mentions them—critical for follow-ups.
- search_queries: YOU decide when to submit a product search. Return search_queries when the user wants product results; return null when you're only clarifying, asking questions, or no search is needed.
  - SUBMIT when: user asks for gift ideas, suggests a product type, or refines a previous request (revision or new topic).
  - DO NOT SUBMIT when: you're only asking a clarifying question, or the message doesn't warrant a product search.
- REVISION vs NEW: Classify each follow-up:
  - REVISION: User adds constraints to the previous request—e.g. "200+", "under $50", "red", "women's only", "budget $100". Keep the product type, add the new constraint. search_queries MUST combine previous context + new constraint (e.g. "red running shoes women under 200").
  - NEW: User asks for something completely different (e.g. "mouse pad" after "red running shoes"). Generate fresh search_queries for the new topic. Ignore previous_search_queries.
- ALWAYS include attributes in search_queries when known:
  - Price: "200+", "under $50", "budget 100" → append "under 200", "under 50", "under 100" to the query string.
  - Gender: wife/mom/daughter → "women"; husband/dad/son → "men". Never return men's products when recipient is female.
  - Color: if user says "red", "blue", etc., include in query (e.g. "red running shoes women").
- Use short, Amazon-style terms: "red running shoes women under 200", "pickleball shoes women".
- subtitle: brief 3-8 word description for the carousel.
- NEVER include search strings in your reply. The reply is user-facing only.
- If unsure about a field, omit or null
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
    # Match **Bold** or - **Bold**: patterns
    for m in re.finditer(r'\*{2}([^*]+)\*{2}', reply or ""):
        phrase = m.group(1).strip()
        if len(phrase) > 2 and len(phrase) < 50 and phrase.lower() not in ("e.g", "etc", "i.e"):
            queries.append(phrase)
    # Also match "- Item:" or "• Item" at line start
    for m in re.finditer(r'^[\s]*[-•*]\s+\*{0,2}([^:*\n]+)', reply or "", re.MULTILINE):
        phrase = m.group(1).strip().strip("*")
        if len(phrase) > 2 and len(phrase) < 50:
            clean = phrase.split(":")[0].strip()
            if clean and clean.lower() not in [q.lower() for q in queries]:
                queries.append(clean)
    return list(dict.fromkeys(queries))[:3]  # dedupe, max 3


def _derive_queries_from_gift_context(gift_context: Optional[dict]) -> list:
    """Derive search queries from gift_context when LLM omits them."""
    if not isinstance(gift_context, dict):
        return []
    queries = []
    for s in (gift_context.get("suggestions_so_far") or [])[:3]:
        if isinstance(s, str) and len(s) > 2:
            clean = s.strip().strip("*").split(":")[0].strip()
            if clean and clean.lower() not in [q.lower() for q in queries]:
                queries.append(clean)
    for i in (gift_context.get("interests") or [])[:2]:
        if isinstance(i, str) and len(i) > 2:
            if i.lower() not in [q.lower() for q in queries]:
                queries.append(i + " gift")
    return queries[:3]


def _is_constraint_revision(message: str) -> bool:
    """True if message looks like a constraint-only refinement (e.g. '200+', 'under $50')."""
    import re
    msg = (message or "").strip().lower()
    if len(msg) > 35:
        return False
    if re.search(r"\d+\s*\+?|\$\s*\d+|\d+\s*\$|under\s*\d+|over\s*\d+|budget\s*\d+|max\s*\d+", msg):
        return True
    if re.match(r"^\d+\+?\s*$", msg) or re.match(r"^\$\d+\s*$", msg):
        return True
    return False


def _inject_gift_context_filters(query: str, gift_context: Optional[dict]) -> str:
    """Inject recipient, budget from gift_context into query if missing."""
    import re
    if not query or not isinstance(gift_context, dict):
        return query
    q = query.lower()
    parts = []
    recipient = (gift_context.get("recipient_info") or "").lower()
    if recipient and "women" not in q and "men" not in q and "womens" not in q and "mens" not in q:
        if any(w in recipient for w in ("wife", "mom", "mother", "daughter", "girlfriend", "her", "woman", "female")):
            parts.append("women")
        elif any(w in recipient for w in ("husband", "dad", "father", "son", "boyfriend", "him", "man", "male")):
            parts.append("men")
    budget = (gift_context.get("budget") or "").strip()
    if budget and "under" not in q and "over" not in q and "budget" not in q and "$" not in q:
        m = re.search(r"\$?\s*(\d+)", budget)
        if m:
            parts.append(f"under {m.group(1)}")
    if not parts:
        return query
    return f"{query.strip()} " + " ".join(parts)


def _merge_constraint_with_previous(message: str, previous_queries: list, gift_context: Optional[dict] = None) -> list:
    """When user sends a constraint revision (e.g. '200+'), merge with previous queries."""
    import re
    if not isinstance(previous_queries, list) or not previous_queries:
        return []
    msg = (message or "").strip()
    constraint = ""
    m = re.search(r"(\d+)\s*\+", msg)
    if m:
        constraint = f"under {m.group(1)}"
    else:
        m = re.search(r"\$\s*(\d+)|\$(\d+)", msg)
        if m:
            constraint = f"under {m.group(1) or m.group(2)}"
        else:
            m = re.search(r"under\s*\$?\s*(\d+)", msg, re.I)
            if m:
                constraint = f"under {m.group(1)}"
            else:
                m = re.search(r"over\s*\$?\s*(\d+)", msg, re.I)
                if m:
                    constraint = f"over {m.group(1)}"
                else:
                    m = re.search(r"budget\s*\$?\s*(\d+)", msg, re.I)
                    if m:
                        constraint = f"under {m.group(1)}"
                    else:
                        m = re.search(r"^(\d+)\s*\+?\s*$", msg)
                        if m:
                            constraint = f"under {m.group(1)}"
    if not constraint:
        return []
    merged = [f"{q.strip()} {constraint}" for q in previous_queries[:3] if str(q).strip()]
    return [_inject_gift_context_filters(m, gift_context) for m in merged]


def _user_message_to_search_queries(message: str) -> list:
    """Last-resort fallback: derive search query from user's message."""
    import re
    msg = (message or "").strip()
    if len(msg) < 5:
        return []
    vague = re.match(r"^(show\s+(me\s+)?more|other\s+options?|more\s+ideas?|anything\s+else)\s*[?!.]?$", msg, re.I)
    if vague:
        return []
    stop = {"for", "my", "the", "a", "an", "to", "and", "or", "with", "around", "about", "budget", "under", "over"}
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
    """Resolve search queries from LLM or fallbacks. Runs even when parsed is None."""
    query_subtitles = []
    if isinstance(parsed, dict):
        sq_list = parsed.get("search_queries")
        if isinstance(sq_list, list) and sq_list:
            query_subtitles = _parse_search_queries(sq_list)
        elif isinstance(parsed.get("search_query"), str):
            sq = (parsed.get("search_query") or "").strip()
            if sq:
                query_subtitles = [(sq, "")]

    if not query_subtitles:
        reply_text = ""
        if isinstance(parsed, dict):
            reply_text = (parsed.get("reply") or "").strip()
        if not reply_text and raw_text:
            reply_text = raw_text.strip()
        queries = _extract_search_queries_from_reply(reply_text)
        if not queries:
            queries = _user_message_to_search_queries(message)
        if not queries and isinstance(previous_queries, list) and previous_queries:
            if _is_constraint_revision(message):
                queries = _merge_constraint_with_previous(message, previous_queries, gift_context)
            if not queries:
                queries = [str(q).strip() for q in previous_queries[:3] if str(q).strip()]
        if not queries and (parsed or gift_context):
            ctx = (parsed.get("gift_context") if isinstance(parsed, dict) else None) or gift_context
            queries = _derive_queries_from_gift_context(ctx)
        query_subtitles = [(q, "") for q in queries if q]
    if query_subtitles and gift_context:
        query_subtitles = [(_inject_gift_context_filters(q, gift_context), s) for q, s in query_subtitles]
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
                raw_results = scrape_amazon_searches(queries, products_per_search=5)
                sub_by_query = {q: s for q, s in query_subtitles}
                products_by_query = [
                    {"query": r["query"], "subtitle": sub_by_query.get(r["query"], ""), "products": r["products"]}
                    for r in raw_results
                ]
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
                    yield _sse("products_loading", {"queries": queries})
                    raw_results = scrape_amazon_searches(queries, products_per_search=5)
                    sub_by_query = {q: s for q, s in query_subtitles}
                    products_by_query = [
                        {"query": r["query"], "subtitle": sub_by_query.get(r["query"], ""), "products": r["products"]}
                        for r in raw_results
                    ]
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
