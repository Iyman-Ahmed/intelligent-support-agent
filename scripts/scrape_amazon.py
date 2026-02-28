"""
scripts/scrape_amazon.py
------------------------
Attempts to scrape public Amazon search results for a given category and
enrich the existing dataset.  Falls back gracefully to the seed data if
Amazon returns a CAPTCHA or bot-detection page.

Usage:
    python scripts/scrape_amazon.py --query "wireless headphones" --pages 3

Requirements (optional — only needed for live scraping):
    pip install requests beautifulsoup4 lxml fake-useragent
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

DATA_FILE = Path(__file__).parent.parent / "data" / "amazon_products.json"

# ---------------------------------------------------------------------------
# Amazon scraper (best-effort, no guarantees against bot detection)
# ---------------------------------------------------------------------------

def _random_headers() -> dict:
    agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.3 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
    ]
    return {
        "User-Agent": random.choice(agents),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
    }


def _is_blocked(html: str) -> bool:
    blocked_signals = [
        "robot check",
        "captcha",
        "automated access",
        "Enter the characters you see below",
        "api-services-support@amazon.com",
    ]
    lower = html.lower()
    return any(s.lower() in lower for s in blocked_signals)


def _parse_search_page(html: str) -> list[dict]:
    """
    Extract basic product info from an Amazon search results page.
    Returns a list of partial product dicts.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("⚠️  beautifulsoup4 not installed. Run: pip install beautifulsoup4 lxml")
        return []

    soup = BeautifulSoup(html, "lxml")
    products = []

    for card in soup.select('[data-component-type="s-search-result"]'):
        try:
            asin = card.get("data-asin", "").strip()
            if not asin:
                continue

            title_el = card.select_one("h2 a span")
            title = title_el.get_text(strip=True) if title_el else "Unknown"

            price_whole = card.select_one(".a-price-whole")
            price_frac  = card.select_one(".a-price-fraction")
            if price_whole:
                price_str = price_whole.get_text(strip=True).replace(",", "")
                frac_str  = price_frac.get_text(strip=True) if price_frac else "00"
                price = float(f"{price_str}.{frac_str}")
            else:
                price = 0.0

            rating_el = card.select_one(".a-icon-alt")
            rating_text = rating_el.get_text(strip=True) if rating_el else "0"
            rating = float(rating_text.split()[0]) if rating_text else 0.0

            review_el = card.select_one('[aria-label*="ratings"]')
            review_text = review_el.get("aria-label", "0").split()[0].replace(",", "") if review_el else "0"
            review_count = int(review_text) if review_text.isdigit() else 0

            brand_el = card.select_one(".a-row.a-size-base.a-color-secondary span")
            brand = brand_el.get_text(strip=True) if brand_el else "Unknown"

            products.append({
                "asin": asin,
                "title": title,
                "brand": brand,
                "category": "Electronics",
                "price": price,
                "rating": rating,
                "review_count": review_count,
                "features": [],
                "description": "",
                "tags": [],
                "reviews": [],
                "qa_pairs": [],
            })
        except Exception:
            continue

    return products


def scrape(query: str = "wireless headphones", pages: int = 2) -> list[dict]:
    """
    Scrape Amazon search results.  Returns scraped products, or [] on failure.
    """
    try:
        import requests
    except ImportError:
        print("⚠️  requests not installed. Run: pip install requests")
        return []

    results: list[dict] = []
    base_url = "https://www.amazon.com/s"

    for page in range(1, pages + 1):
        params = {"k": query, "page": page}
        try:
            resp = requests.get(
                base_url,
                params=params,
                headers=_random_headers(),
                timeout=15,
            )
        except Exception as exc:
            print(f"  Network error on page {page}: {exc}")
            break

        if resp.status_code != 200 or _is_blocked(resp.text):
            print(f"  ⚠️  Blocked or non-200 on page {page} (status={resp.status_code}). Stopping.")
            break

        page_products = _parse_search_page(resp.text)
        results.extend(page_products)
        print(f"  Page {page}: found {len(page_products)} products.")

        # Polite crawl delay
        time.sleep(random.uniform(3.0, 6.0))

    return results


# ---------------------------------------------------------------------------
# Merge with existing dataset
# ---------------------------------------------------------------------------

def merge_into_dataset(new_products: list[dict], data_file: Path = DATA_FILE) -> None:
    """Merge scraped products into existing JSON, skipping duplicate ASINs."""
    with open(data_file, "r", encoding="utf-8") as f:
        existing: list[dict] = json.load(f)

    existing_asins = {p["asin"] for p in existing}
    added = 0
    for p in new_products:
        if p["asin"] and p["asin"] not in existing_asins:
            existing.append(p)
            existing_asins.add(p["asin"])
            added += 1

    if added:
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        print(f"✅  Added {added} new products to {data_file}")
    else:
        print("ℹ️  No new products to add (all duplicates or empty result).")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Amazon product data.")
    parser.add_argument("--query",  default="wireless headphones", help="Search query")
    parser.add_argument("--pages",  type=int, default=2, help="Pages to scrape (default 2)")
    parser.add_argument("--dry-run", action="store_true", help="Print results without saving")
    args = parser.parse_args()

    print(f"🔍  Scraping Amazon for: \"{args.query}\" ({args.pages} page(s))…")
    products = scrape(query=args.query, pages=args.pages)

    if not products:
        print("⚠️  No products scraped (bot-detection or missing deps). "
              "The existing seed dataset will be used as-is.")
        return

    print(f"\n📦  Scraped {len(products)} products total.")

    if args.dry_run:
        print(json.dumps(products[:3], indent=2))
        return

    merge_into_dataset(products)


if __name__ == "__main__":
    main()
