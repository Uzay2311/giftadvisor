"""
Amazon product search via SerpAPI Amazon engine.
Submits 2-3 different search queries, returns structured products for carousel.
"""
import os
from typing import List
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from serpapi import GoogleSearch
except ImportError:
    # Compatibility import path used by some google-search-results versions.
    from serpapi.google_search import GoogleSearch

DEFAULT_TAG = "bestgift0514-20"


def _add_affiliate_tag(url: str) -> str:
    """Add Amazon affiliate tag to product URL (&tag=bestgift0514-20)."""
    if not url or "amazon." not in url.lower():
        return url
    tag = os.getenv("AMAZON_AFFILIATE_TAG", DEFAULT_TAG).strip() or DEFAULT_TAG
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    qs["tag"] = [tag]
    new_query = urlencode(qs, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def _serpapi_amazon_search(query: str, max_results: int = 5) -> List[dict]:
    """Fetch Amazon products via SerpAPI Amazon engine only."""
    api_key = os.getenv("SERPAPI_API_KEY", "").strip()
    if not api_key:
        return []
    try:
        search = GoogleSearch(
            {
                "api_key": api_key,
                "engine": "amazon",
                "k": query,
                "amazon_domain": "amazon.com",
                "device": "desktop",
            }
        )
        data = search.get_dict()
    except Exception as e:
        print("SerpAPI Amazon error:", type(e).__name__)
        return []
    products = []
    # Use organic results only; skip sponsored products explicitly.
    items = data.get("organic_results") or []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("sponsored") is True:
            continue
        link = (item.get("link_clean") or item.get("link") or "").strip()
        if not link or "amazon." not in link.lower():
            continue
        link = _add_affiliate_tag(link)
        price_val = item.get("price") or item.get("extracted_price")
        if isinstance(price_val, (int, float)):
            price_str = f"${price_val:.2f}" if price_val else ""
        else:
            price_str = str(price_val or "").strip()
        rating = item.get("rating")
        reviews = item.get("reviews")
        products.append({
            "title": (item.get("title") or "").strip(),
            "link": link,
            "image": (item.get("thumbnail") or item.get("image") or "").strip(),
            "price": price_str,
            "rating": rating,
            "reviews": reviews,
            "bought_last_month": item.get("bought_last_month"),
        })
        if len(products) >= max_results:
            break
    return products


def scrape_amazon_searches(queries: List[str], products_per_search: int = 3) -> List[dict]:
    """
    Submit up to 3 search queries. Returns list of {query, products} per query.
    Uses SerpAPI only.
    Product links include &tag=bestgift0514-20.
    """
    if not queries or not isinstance(queries, list):
        return []
    clean_queries = [str(q or "").strip() for q in queries[:3] if str(q or "").strip()]
    if not clean_queries:
        return []
    def _fetch_for_query(q: str) -> List[dict]:
        seen = set()
        items = _serpapi_amazon_search(q, max_results=products_per_search)

        filtered = [i for i in items if (i.get("title") or "").strip().lower() not in seen]
        for i in filtered:
            seen.add((i.get("title") or "").strip().lower())
        return filtered

    # Fetch each query in parallel; preserve original query order in output.
    max_workers = min(4, len(clean_queries)) if clean_queries else 1
    out_by_idx = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_for_query, q): (idx, q) for idx, q in enumerate(clean_queries)}
        for fut in as_completed(futures):
            idx, q = futures[fut]
            try:
                filtered = fut.result() or []
            except Exception as e:
                print("Parallel query fetch error:", type(e).__name__)
                filtered = []
            if filtered:
                out_by_idx[idx] = {"query": q, "products": filtered}

    return [out_by_idx[i] for i in sorted(out_by_idx.keys())]
