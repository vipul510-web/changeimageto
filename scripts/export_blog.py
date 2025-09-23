#!/usr/bin/env python3
"""
Export dynamic blog pages from production backend to static files under frontend/blog/.

Strategy:
- Fetch the blog index HTML from BACKEND_BLOG_URL (/blog)
- Parse all /blog/*.html links
- Download each article HTML and save it as frontend/blog/<slug>.html
- Also save a copy of the fetched index as frontend/blog/index.html (so it stays in sync)

Usage:
  python scripts/export_blog.py

Env (optional):
  BACKEND_BLOG_URL   default: https://bgremover-backend-121350814881.us-central1.run.app/blog

Requirements: requests (pip install requests)
"""
import os
import re
import sys
import time
from urllib.parse import urljoin

try:
    import requests
except Exception:
    print("This script requires 'requests'. Install with: pip install requests", file=sys.stderr)
    sys.exit(1)


BACKEND_BLOG_URL = os.environ.get(
    "BACKEND_BLOG_URL",
    "https://bgremover-backend-121350814881.us-central1.run.app/blog",
)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT_DIR, "frontend", "blog")


def ensure_out_dir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


def fetch(url: str) -> str:
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    # Ensure text encoding
    resp.encoding = resp.apparent_encoding or resp.encoding or "utf-8"
    return resp.text


def extract_slugs(index_html: str) -> list:
    # Match both single and double quoted hrefs; allow relative or absolute URLs
    pattern = r"href=([\'\"])((?:https?:\/\/[^\"\']+)?\/blog\/([a-z0-9\-]+)\.html)\1"
    hrefs = re.findall(pattern, index_html, flags=re.I)
    slugs = []
    seen = set()
    for _quote, full, slug in hrefs:
        if slug not in seen:
            seen.add(slug)
            slugs.append(slug)
    return slugs


def write_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main() -> int:
    ensure_out_dir()
    print(f"Fetching blog index: {BACKEND_BLOG_URL}")
    index_html = fetch(BACKEND_BLOG_URL)
    # Save index as-is
    write_file(os.path.join(OUT_DIR, "index.html"), index_html)
    print("Saved: frontend/blog/index.html")

    # Extract slugs and fetch each
    slugs = extract_slugs(index_html)
    if not slugs:
        print("No blog article links found on index; nothing else to export.")
        return 0
    print(f"Found {len(slugs)} article(s): {', '.join(slugs)}")
    base = BACKEND_BLOG_URL.rstrip("/") + "/"
    ok = 0
    for slug in slugs:
        url = urljoin(base, f"{slug}.html")
        try:
            html = fetch(url)
            write_file(os.path.join(OUT_DIR, f"{slug}.html"), html)
            ok += 1
            print(f"Saved: frontend/blog/{slug}.html")
            time.sleep(0.2)  # be polite
        except Exception as e:
            print(f"WARN: failed to fetch {url}: {e}")

    print(f"Export complete. {ok}/{len(slugs)} articles saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


