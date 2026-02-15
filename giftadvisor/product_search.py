"""
Amazon product search: SerpAPI Amazon (primary), Serper Shopping, or ScraperAPI scraping.
Submits 2-3 different search queries, returns structured products for carousel.
"""
import os
import requests
from typing import List
from urllib.parse import quote_plus, urlparse, parse_qs, urlencode, urlunparse

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

AMAZON_BASE = "https://www.amazon.com/s"
DEFAULT_TAG = "bestgift0514-20"
SCRAPER_API = "http://api.scraperapi.com"
SERPER_SHOPPING = "https://google.serper.dev/shopping"
SERPAPI_SEARCH = "https://serpapi.com/search.json"


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


def build_amazon_search_url(query: str) -> str:
    """Build Amazon search URL from query."""
    q = (query or "").strip()
    if not q:
        return ""
    tag = os.getenv("AMAZON_AFFILIATE_TAG", DEFAULT_TAG).strip() or DEFAULT_TAG
    encoded = quote_plus(q)
    return f"{AMAZON_BASE}?k={encoded}&tag={tag}"


def _fetch_page(url: str) -> str:
    """Fetch page HTML via ScraperAPI or direct request."""
    api_key = os.getenv("SCRAPER_API_KEY", "").strip()
    if api_key:
        resp = requests.get(
            SCRAPER_API,
            params={"api_key": api_key, "url": url},
            timeout=30,
        )
    else:
        resp = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=15,
        )
    resp.raise_for_status()
    return resp.text


def _parse_amazon_search_results(html: str, max_per_page: int = 5) -> List[dict]:
    """Parse Amazon search results HTML into product dicts."""
    if not BeautifulSoup or not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    products = []
    # Amazon search result cards (try multiple selectors)
    cards = soup.select('[data-component-type="s-search-result"]')
    if not cards:
        cards = soup.select('div[data-asin]:not([data-asin=""])')
    if not cards:
        cards = soup.select(".s-result-item")
    cards = cards[:max_per_page]
    for card in cards:
        try:
            # Skip template placeholders
            asin = card.get("data-asin", "")
            if asin and asin.startswith("slot-"):
                continue
            # Skip sponsored if desired
            if card.select_one(".s-sponsored-label-info-icon"):
                continue
            title_el = card.select_one("h2 a.a-link-normal") or card.select_one("h2 a")
            title = title_el.get_text(strip=True) if title_el else ""
            link = ""
            if title_el and title_el.get("href"):
                href = title_el["href"]
                if href.startswith("/"):
                    link = "https://www.amazon.com" + href.split("?")[0]
                else:
                    link = href.split("?")[0]
                link = _add_affiliate_tag(link)
            img_el = card.select_one("img.s-image")
            image = img_el.get("src", "") if img_el else ""
            price_el = card.select_one(".a-price .a-offscreen")
            price = price_el.get_text(strip=True) if price_el else ""
            if not price:
                price_whole = card.select_one(".a-price-whole")
                price_frac = card.select_one(".a-price-fraction")
                if price_whole:
                    price = price_whole.get_text(strip=True) or ""
                    if price_frac:
                        price += price_frac.get_text(strip=True) or ""
                    if price and not price.startswith("$"):
                        price = "$" + price
            rating_el = card.select_one(".a-icon-alt")
            rating = rating_el.get_text(strip=True) if rating_el else ""
            desc_el = card.select_one(".a-color-secondary.a-size-base")
            description = desc_el.get_text(strip=True)[:200] if desc_el else ""
            if title and (link or image):
                products.append({
                    "title": title,
                    "link": link,
                    "image": image,
                    "price": price,
                    "description": description or rating,
                })
        except Exception:
            continue
    return products


def _serpapi_amazon_search(query: str, max_results: int = 5) -> List[dict]:
    """Fetch Amazon products via SerpAPI Amazon engine. Returns native Amazon URLs."""
    api_key = os.getenv("SERPAPI_API_KEY", "").strip()
    if not api_key:
        return []
    try:
        resp = requests.get(
            SERPAPI_SEARCH,
            params={
                "api_key": api_key,
                "engine": "amazon",
                "k": query,
                "amazon_domain": "amazon.com",
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print("SerpAPI Amazon error:", type(e).__name__)
        return []
    products = []
    # organic_results + product_ads.products
    items = []
    for r in data.get("organic_results") or []:
        items.append(r)
    ads = data.get("product_ads") or {}
    for p in ads.get("products") or []:
        items.append(p)
    for item in items[:max_results]:
        if not isinstance(item, dict):
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
        desc = f"{rating} stars, {reviews} reviews" if (rating and reviews) else ""
        products.append({
            "title": (item.get("title") or "").strip(),
            "link": link,
            "image": (item.get("thumbnail") or item.get("image") or "").strip(),
            "price": price_str,
            "description": desc,
        })
    return products


def _serper_search(query: str, max_results: int = 5, amazon_only: bool = True) -> List[dict]:
    """Fetch products via Serper Shopping API. Filter to Amazon URLs only."""
    api_key = os.getenv("SERPER_API_KEY", "").strip()
    if not api_key:
        return []
    try:
        # Add "amazon" to query to bias toward Amazon product results
        search_q = f"{query} amazon" if amazon_only else query
        resp = requests.post(
            SERPER_SHOPPING,
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": search_q, "gl": "us", "hl": "en"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print("Serper search error:", repr(e))
        return []
    products = []
    shopping = data.get("shopping") or data.get("products") or []
    for item in shopping:
        if not isinstance(item, dict):
            continue
        link = (item.get("link") or item.get("productLink") or item.get("url") or "").strip()
        # Only include Amazon URLs
        if not link or "amazon." not in link.lower():
            continue
        link = _add_affiliate_tag(link)
        price_val = item.get("price") or item.get("extractedPrice")
        if isinstance(price_val, (int, float)):
            price_str = f"${price_val:.2f}" if price_val else ""
        else:
            price_str = str(price_val or "").strip()
        products.append({
            "title": (item.get("title") or item.get("name") or "").strip(),
            "link": link,
            "image": (item.get("image") or item.get("thumbnail") or item.get("imageUrl") or "").strip(),
            "price": price_str,
            "description": (item.get("snippet") or item.get("description") or item.get("body") or "").strip()[:200],
        })
        if len(products) >= max_results:
            break
    return products


def scrape_amazon_searches(queries: List[str], products_per_search: int = 3) -> List[dict]:
    """
    Submit 2-3 search queries. Returns list of {query, products} per query.
    Tries SerpAPI first; falls back to Serper then ScraperAPI if empty.
    Product links include &tag=bestgift0514-20.
    """
    if not queries or not isinstance(queries, list):
        return []
    clean_queries = [str(q or "").strip() for q in queries[:3] if str(q or "").strip()]
    if not clean_queries:
        return []

    # Prefer SerpAPI Amazon
    if os.getenv("SERPAPI_API_KEY"):
        out = []
        seen = set()
        for q in clean_queries:
            items = _serpapi_amazon_search(q, max_results=products_per_search)
            filtered = [i for i in items if (i.get("title") or "").strip().lower() not in seen]
            for i in filtered:
                seen.add((i.get("title") or "").strip().lower())
            if filtered:
                out.append({"query": q, "products": filtered})
        if out:
            return out

    # Fallback: Serper Shopping
    if os.getenv("SERPER_API_KEY"):
        out = []
        seen = set()
        for q in clean_queries:
            items = _serper_search(q, max_results=products_per_search, amazon_only=True)
            filtered = [i for i in items if (i.get("title") or "").strip().lower() not in seen]
            for i in filtered:
                seen.add((i.get("title") or "").strip().lower())
            if filtered:
                out.append({"query": q, "products": filtered})
        if out:
            return out

    # Fallback: Scrape Amazon via ScraperAPI
    if os.getenv("SCRAPER_API_KEY"):
        out = []
        seen = set()
        for q in clean_queries:
            url = build_amazon_search_url(q)
            if not url:
                continue
            try:
                html = _fetch_page(url)
                items = _parse_amazon_search_results(html, max_per_page=products_per_search)
                filtered = [i for i in items if (i.get("title") or "").strip().lower() not in seen]
                for i in filtered:
                    seen.add((i.get("title") or "").strip().lower())
                if filtered:
                    out.append({"query": q, "products": filtered})
            except requests.exceptions.HTTPError as e:
                if e.response and e.response.status_code == 401:
                    print("ScraperAPI 401: Invalid key or no credits. Check SCRAPER_API_KEY.")
                else:
                    print("Amazon scrape error:", e.response.status_code if e.response else repr(e))
            except Exception as e:
                print("Amazon scrape error:", type(e).__name__)
        if out:
            return out

    return []
