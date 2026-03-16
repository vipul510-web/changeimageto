"""
Analyze GA4 + GSC data and generate SEO tactical ideas.
Includes: conflicts, low-hanging fruit, on-page checks, keyword clusters, competitor gaps.
"""
import re
from pathlib import Path
from collections import defaultdict

# Map keywords/phrases to relevant pages (for internal linking & content)
KEYWORD_TO_PAGE = {
    "bulk resize": "/bulk-image-resizer.html",
    "batch resize": "/bulk-image-resizer.html",
    "bulk image resizer": "/bulk-image-resizer.html",
    "batch image resize": "/bulk-image-resizer.html",
    "bulk resize photos": "/bulk-image-resizer.html",
    "batch resize photos": "/bulk-image-resizer.html",
    "bulk image resizing": "/bulk-image-resizer.html",
    "bulk photo resize": "/bulk-image-resizer.html",
    "bulkresize": "/bulk-image-resizer.html",
    "blur background": "/blur-background.html",
    "blur background photo": "/blur-background.html",
    "canva background remover": "/remove-background-from-image.html",
    "canva remove background": "/remove-background-from-image.html",
    "change background color": "/change-image-background.html",
    "change background color online": "/change-image-background.html",
    "blue background": "/change-image-background-to-blue.html",
    "black background": "/change-image-background-to-black.html",
    "white background": "/change-image-background-to-white.html",
    "amazon product image": "/remove-background-for-amazon.html",
    "amazon white background": "/remove-background-for-amazon.html",
    "remove background": "/remove-background-from-image.html",
    "background remover": "/remove-background-from-image.html",
}

# Stopwords for clustering (ignore common words)
STOPWORDS = {"the", "a", "an", "to", "for", "of", "in", "on", "how", "free", "online", "image", "photo", "photos"}


def normalize_path(path: str) -> str:
    """Normalize path for matching."""
    if not path.startswith("/"):
        path = "/" + path
    if path.endswith("/") and path != "/":
        path = path.rstrip("/")
    return path.replace("/index.html", "/")


def find_best_page_for_keyword(keyword: str) -> str | None:
    """Find the best page to target for a keyword."""
    kw_lower = keyword.lower()
    for phrase, page in KEYWORD_TO_PAGE.items():
        if phrase in kw_lower:
            return page
    return None


# ---------------------------------------------------------------------------
# 1. Keyword conflict / cannibalization detection
# ---------------------------------------------------------------------------
def detect_keyword_conflicts(gsc_query_page: list[dict], exclude_patterns: list[str]) -> list[dict]:
    """
    Find queries where multiple pages compete. Intent-level conflict: same query,
    different URLs. Returns recommendations: which URL should win, which to merge/redirect.
    """
    exclude = set(exclude_patterns or [])
    # Group by query: {query: [(path, impressions, position, clicks), ...]}
    by_query = defaultdict(list)
    for row in gsc_query_page:
        path = row["path"]
        if any(pat in path for pat in exclude):
            continue
        if row["impressions"] < 5:  # Ignore noise
            continue
        by_query[row["query"]].append({
            "path": path,
            "impressions": row["impressions"],
            "position": row["position"],
            "clicks": row["clicks"],
        })

    conflicts = []
    for query, pages in by_query.items():
        if len(pages) < 2:
            continue
        # Sort by best performance: higher impressions, better position, more clicks
        pages_sorted = sorted(
            pages,
            key=lambda p: (p["impressions"], -p["clicks"], -1 / max(p["position"], 0.1)),
            reverse=True,
        )
        winner = pages_sorted[0]
        losers = pages_sorted[1:]
        conflicts.append({
            "query": query,
            "winner": winner["path"],
            "winner_impressions": winner["impressions"],
            "losers": [{"path": p["path"], "impressions": p["impressions"], "action": "merge_or_redirect"} for p in losers],
            "recommendation": f"Consolidate: {winner['path']} should win. Consider 301 redirect from {losers[0]['path']}.",
        })
    return sorted(conflicts, key=lambda c: c["winner_impressions"], reverse=True)[:15]


# ---------------------------------------------------------------------------
# 2. Low-hanging fruit: position 31+ (page 4+) ripe for top 1-3
# ---------------------------------------------------------------------------
def find_low_hanging_fruit(gsc_keywords: list[dict], exclude_patterns: list[str]) -> list[dict]:
    """
    Keywords at position 31+ (page 4+) with decent impressions. Small optimizations
    can push them into top 1-3 pages.
    """
    exclude = set(exclude_patterns or [])
    ripe = []
    for kw in gsc_keywords:
        if kw["position"] < 31 or kw["impressions"] < 20:
            continue
        page = find_best_page_for_keyword(kw["query"])
        if not page or any(pat in page for pat in exclude):
            continue
        ripe.append({
            "type": "low_hanging_fruit",
            "priority": "medium",
            "keyword": kw["query"],
            "position": kw["position"],
            "impressions": kw["impressions"],
            "clicks": kw["clicks"],
            "target_page": page,
            "action": "add_keyword_to_meta",
            "description": f"'{kw['query']}' at position {kw['position']} (page {kw['position']//10 + 1}) with {kw['impressions']} impr. Optimize {page} to push into top 3.",
        })
    return sorted(ripe, key=lambda x: (x["impressions"], -x["position"]), reverse=True)[:15]


# ---------------------------------------------------------------------------
# 3. On-page check: top queries in title, meta description, H1
# ---------------------------------------------------------------------------
def _extract_page_seo(html_path: Path) -> dict:
    """Extract title, meta description, H1 from HTML."""
    try:
        content = html_path.read_text(encoding="utf-8")
    except Exception:
        return {"title": "", "description": "", "h1": ""}

    title = ""
    m = re.search(r"<title[^>]*>([^<]+)</title>", content, re.I)
    if m:
        title = m.group(1).strip()

    desc = ""
    m = re.search(r'<meta\s+name=["\']description["\']\s+content=["\']([^"\']*)["\']', content, re.I)
    if m:
        desc = m.group(1).strip()

    h1 = ""
    m = re.search(r"<h1[^>]*>([^<]+)</h1>", content, re.I)
    if m:
        h1 = m.group(1).strip()

    return {"title": title, "description": desc, "h1": h1}


def _keyword_in_text(keyword: str, text: str) -> bool:
    """Check if keyword (or significant part) appears in text (case-insensitive)."""
    if not text:
        return False
    kw_lower = keyword.lower()
    text_lower = text.lower()
    # Exact phrase or all significant words present
    if kw_lower in text_lower:
        return True
    words = [w for w in kw_lower.split() if len(w) > 2 and w not in STOPWORDS]
    return all(w in text_lower for w in words) if words else False


def check_onpage_optimization(
    gsc_data: dict,
    gsc_query_page: list[dict],
    frontend_dir: str,
    project_root: Path,
    exclude_patterns: list[str],
) -> list[dict]:
    """
    Check if top queries appear in title, meta description, H1.
    Returns gaps: {path, query, missing_in: [title|description|h1], impressions}.
    """
    exclude = set(exclude_patterns or [])
    frontend = project_root / frontend_dir

    # Get top queries per page from query+page data
    page_queries = defaultdict(list)  # path -> [(query, impressions), ...]
    for row in gsc_query_page:
        path = normalize_path(row["path"])
        if any(pat in path for pat in exclude) or row["impressions"] < 30:
            continue
        page_queries[path].append((row["query"], row["impressions"]))

    # Also add top keywords from global list for pages we know
    for kw in gsc_data["keywords"][:100]:
        if kw["impressions"] < 30:
            continue
        page = find_best_page_for_keyword(kw["query"])
        if page:
            path = normalize_path(page)
            if path not in page_queries or not any(q[0] == kw["query"] for q in page_queries[path]):
                page_queries[path].append((kw["query"], kw["impressions"]))

    gaps = []
    for path, queries in page_queries.items():
        # Resolve path to file
        if path == "/":
            html_path = frontend / "index.html"
        else:
            rel = path.lstrip("/")
            html_path = frontend / rel
            if not html_path.suffix:
                html_path = html_path.with_suffix(".html")
            if not html_path.exists() and (frontend / rel / "index.html").exists():
                html_path = frontend / rel / "index.html"
        if not html_path.exists():
            continue

        seo = _extract_page_seo(html_path)
        # Sort by impressions, take top 5 per page
        for query, imp in sorted(queries, key=lambda x: -x[1])[:5]:
            missing = []
            if not _keyword_in_text(query, seo["title"]):
                missing.append("title")
            if not _keyword_in_text(query, seo["description"]):
                missing.append("description")
            if not _keyword_in_text(query, seo["h1"]):
                missing.append("h1")
            if missing:
                gaps.append({
                    "type": "onpage_gap",
                    "priority": "high" if "title" in missing or "h1" in missing else "medium",
                    "path": path,
                    "query": query,
                    "impressions": imp,
                    "missing_in": missing,
                    "target_page": path,
                    "action": "fix_onpage_gaps",
                    "description": f"'{query}' ({imp} impr) missing from {', '.join(missing)} on {path}",
                })

    return sorted(gaps, key=lambda x: (-x["impressions"], len(x["missing_in"])))[:20]


# ---------------------------------------------------------------------------
# 4. Keyword clustering and aggregate performance
# ---------------------------------------------------------------------------
def _cluster_key(words: list[str]) -> str:
    """Create a cluster key from significant words (sorted)."""
    significant = sorted(w for w in words if len(w) > 2 and w not in STOPWORDS)
    return " ".join(significant[:4]) if significant else "other"


def cluster_keywords(gsc_keywords: list[dict]) -> list[dict]:
    """
    Group related keywords into clusters. Track aggregate impressions/clicks.
    Identifies which topics drive traffic and where clusters are thin.
    """
    clusters = defaultdict(lambda: {"keywords": [], "impressions": 0, "clicks": 0})
    for kw in gsc_keywords:
        words = kw["query"].lower().split()
        key = _cluster_key(words)
        clusters[key]["keywords"].append({"query": kw["query"], "impressions": kw["impressions"], "clicks": kw["clicks"]})
        clusters[key]["impressions"] += kw["impressions"]
        clusters[key]["clicks"] += kw["clicks"]

    result = []
    for name, data in clusters.items():
        if data["impressions"] < 10:
            continue
        result.append({
            "cluster": name,
            "keywords": sorted(data["keywords"], key=lambda x: -x["impressions"])[:10],
            "total_impressions": data["impressions"],
            "total_clicks": data["clicks"],
            "keyword_count": len(data["keywords"]),
        })
    return sorted(result, key=lambda x: -x["total_impressions"])[:20]


# ---------------------------------------------------------------------------
# 5. Competitor gap analysis (optional, requires SerpAPI)
# ---------------------------------------------------------------------------
def analyze_competitor_gaps(
    config: dict,
    gsc_keywords: list[dict],
    exclude_patterns: list[str] | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Analyze SERP pages 1-3 (positions 1-30) for target keywords.
    Returns (gaps_insights, tactics, serp_related_keywords) where:
    - gaps_insights: full report for email (our position, top competitors with titles/snippets, gap analysis)
    - tactics: implementable tactics (add competitor keywords to our pages when we rank 4-10)
    - serp_related_keywords: NEW keywords from "Related searches" (no GSC presence) for blog creation
    """
    import os
    api_key = config.get("serp_api_key") or os.environ.get("SERP_API_KEY")
    if not api_key:
        return [], [], []

    try:
        import requests
    except ImportError:
        return [], [], []

    exclude = set(exclude_patterns or [])
    gaps = []
    tactics = []
    serp_related = []  # NEW keywords from "Related searches" - we may have zero GSC presence
    gsc_queries = {kw["query"].lower() for kw in gsc_keywords}
    site_topics = {"remove", "background", "resize", "blur", "change", "image", "photo", "canva", "photoshop", "figma", "free", "online", "tool", "quality", "checker", "bulk", "batch"}
    max_kws = config.get("max_competitor_keywords", 12)
    # Top keywords by impressions to check SERP
    top_kws = [kw for kw in gsc_keywords if kw["impressions"] >= 30][:max_kws]

    for kw in top_kws:
        try:
            r = requests.get(
                "https://serpapi.com/search",
                params={"q": kw["query"], "api_key": api_key, "num": 30, "engine": "google"},
                timeout=15,
            )
            if r.status_code != 200:
                continue
            data = r.json()
            organic = data.get("organic_results", [])
            # Find our position and top 3 competitors (pages 1-3 = positions 1-9)
            our_pos = None
            our_url = None
            top3 = []
            for i, res in enumerate(organic[:30]):
                pos = i + 1
                url = res.get("link", "")
                if "changeimageto.com" in url:
                    our_pos = pos
                    our_url = url
                elif pos <= 9:  # Pages 1-3
                    domain = url.split("/")[2] if "/" in url else url
                    try:
                        domain = domain.replace("www.", "").split(".")[-2] + "." + domain.split(".")[-1]
                    except Exception:
                        pass
                    top3.append({
                        "position": pos,
                        "domain": domain,
                        "title": res.get("title", ""),
                        "snippet": (res.get("snippet") or res.get("description") or "")[:150],
                    })

            # Build insight
            insight = {
                "query": kw["query"],
                "impressions": kw["impressions"],
                "clicks": kw["clicks"],
                "our_position": our_pos,
                "our_url": our_url,
                "top3_competitors": top3[:3],
                "on_page1": our_pos is not None and our_pos <= 3,
                "on_page2_3": our_pos is not None and 4 <= our_pos <= 9,
                "not_ranking": our_pos is None or our_pos > 9,
            }

            # Recommendation
            if insight["not_ranking"]:
                insight["recommendation"] = f"Not in top 30. Create blog post or page targeting '{kw['query']}'."
            elif insight["on_page2_3"]:
                insight["recommendation"] = f"On page {((our_pos or 1) - 1) // 3 + 1}. Improve title/meta to match top 3 and push to page 1."
            else:
                insight["recommendation"] = "On page 1. Maintain and optimize for featured snippets."

            gaps.append(insight)

            # Extract "Related searches" - NEW keywords we may have zero presence for
            for rel in data.get("related_searches", [])[:5]:
                q = (rel.get("query") or "").strip()
                if not q or len(q) < 4:
                    continue
                q_lower = q.lower()
                if q_lower in gsc_queries:
                    continue  # We already have this in GSC
                words = set(re.findall(r"\b[a-z0-9]{2,}\b", q_lower)) - STOPWORDS
                if not (words & site_topics):
                    continue  # Not relevant to our site
                serp_related.append({"query": q, "source": "serp_related", "parent_query": kw["query"]})

            # Generate implementable tactic if we rank 4-15 and have a target page
            target_page = find_best_page_for_keyword(kw["query"])
            if target_page and our_pos and 4 <= our_pos <= 15 and top3 and not any(pat in target_page for pat in exclude):
                # Extract keywords from top competitor titles that we might add
                competitor_titles = " ".join(c["title"] for c in top3).lower()
                query_words = set(kw["query"].lower().split())
                # Find significant words in competitor titles that aren't in our query
                competitor_words = set(re.findall(r"\b[a-z0-9]{3,}\b", competitor_titles))
                add_words = competitor_words - query_words - STOPWORDS
                if add_words:
                    # Pick 1-2 most relevant (e.g. "free", "online", "tool" - but not generic)
                    priority = ["free", "online", "tool", "best", "easy", "remove", "background", "resize", "blur"]
                    add_candidates = [w for w in add_words if w in priority]
                    if not add_candidates:
                        add_candidates = list(add_words)[:2]
                    if add_candidates:
                        add_kw = " ".join(add_candidates[:2])
                        tactics.append({
                            "type": "competitor_gap",
                            "priority": "medium",
                            "keyword": add_kw,
                            "query": kw["query"],
                            "our_position": our_pos,
                            "impressions": kw["impressions"],
                            "target_page": target_page,
                            "action": "add_keyword_to_meta",
                            "description": f"Competitor gap: '{kw['query']}' we rank #{our_pos}. Top 3 use '{add_kw}'. Add to {target_page}.",
                        })
        except Exception:
            continue
    # Dedupe serp_related by query
    seen_rel = set()
    serp_related_deduped = []
    for r in serp_related:
        if r["query"].lower() not in seen_rel:
            seen_rel.add(r["query"].lower())
            serp_related_deduped.append(r)
    return gaps[:12], tactics[:5], serp_related_deduped[:15]


# ---------------------------------------------------------------------------
# Main: generate tactics and insights
# ---------------------------------------------------------------------------
def generate_tactics(
    ga4_data: dict,
    gsc_data: dict,
    decliners: list,
    gsc_query_page: dict | None = None,
    config: dict | None = None,
) -> dict:
    """
    Combine all analyses. Returns:
    { tactics: [...], insights: { conflicts, low_hanging, onpage_gaps, clusters, competitor_gaps } }
    """
    config = config or {}
    exclude = config.get("exclude_patterns", [])
    frontend = config.get("frontend_dir", "frontend")
    project_root = Path(config.get("project_root", ".")).resolve()
    if not project_root.is_absolute():
        project_root = (Path(__file__).parent.parent.parent / project_root).resolve()

    tactics = []
    insights = {
        "conflicts": [],
        "low_hanging": [],
        "onpage_gaps": [],
        "clusters": [],
        "competitor_gaps": [],
    }

    # 1. Keyword conflicts (report only)
    if gsc_query_page and gsc_query_page.get("query_page"):
        insights["conflicts"] = detect_keyword_conflicts(gsc_query_page["query_page"], exclude)

    # 2. Low-hanging fruit (becomes tactics)
    insights["low_hanging"] = find_low_hanging_fruit(gsc_data["keywords"], exclude)
    tactics.extend(insights["low_hanging"])

    # 3. On-page gaps (becomes tactics)
    if gsc_query_page and gsc_query_page.get("query_page"):
        insights["onpage_gaps"] = check_onpage_optimization(
            gsc_data, gsc_query_page["query_page"], frontend, project_root, exclude
        )
        tactics.extend(insights["onpage_gaps"])

    # 4. Keyword clusters (report only)
    insights["clusters"] = cluster_keywords(gsc_data["keywords"])

    # 5. Competitor gaps (pages 1-3 SERP analysis, optional)
    comp_gaps, comp_tactics, serp_related = analyze_competitor_gaps(config, gsc_data["keywords"], exclude)
    insights["competitor_gaps"] = comp_gaps
    insights["serp_related_keywords"] = serp_related  # NEW keywords from "Related searches" (no GSC presence)
    tactics.extend(comp_tactics)

    # 6. Declining keywords (existing)
    for d in decliners[:15]:
        if d["imp_drop"] < 5:
            continue
        page = find_best_page_for_keyword(d["query"])
        if page and not any(pat in page for pat in exclude):
            tactics.append({
                "type": "declining_keyword",
                "priority": "high" if d["imp_drop"] > 30 else "medium",
                "keyword": d["query"],
                "imp_drop": d["imp_drop"],
                "target_page": page,
                "action": "add_keyword_to_meta",
                "description": f"Keyword '{d['query']}' lost {d['imp_drop']} impressions. Add to meta/keywords on {page}",
            })

    # 7. High-impression, low-CTR (existing)
    for kw in gsc_data["keywords"]:
        if kw["impressions"] < 50 or kw["ctr"] > 3:
            continue
        pos = kw["position"]
        if 5 <= pos <= 20 and kw["clicks"] < 5:
            page = find_best_page_for_keyword(kw["query"])
            if page and not any(pat in page for pat in exclude):
                tactics.append({
                    "type": "low_ctr_opportunity",
                    "priority": "medium",
                    "keyword": kw["query"],
                    "impressions": kw["impressions"],
                    "ctr": kw["ctr"],
                    "position": kw["position"],
                    "target_page": page,
                    "action": "improve_meta_description",
                    "description": f"'{kw['query']}' has {kw['impressions']} impr, {kw['ctr']}% CTR at pos {pos}. Improve meta to boost CTR.",
                })

    # 8. Traffic vs search gap (existing)
    ga4_paths = {normalize_path(p["path"]): p for p in ga4_data["pages"]}
    gsc_paths = {normalize_path(p["path"]): p for p in gsc_data["pages"]}
    for path, ga4_p in list(ga4_paths.items())[:20]:
        if ga4_p["page_views"] < 50:
            continue
        gsc_p = gsc_paths.get(path, {})
        gsc_imp = gsc_p.get("impressions", 0)
        if gsc_imp < 20 and ga4_p["page_views"] > 100 and not any(pat in path for pat in exclude):
            # Source pages: where to add links FROM. Index and blog for tool pages; blog only for homepage.
            if path == "/":
                source_pages = ["/blog/"]  # Homepage: add links from blog to homepage
            else:
                source_pages = ["/", "/blog/"]  # Tool page: add from index and blog
            tactics.append({
                "type": "traffic_vs_search_gap",
                "priority": "low",
                "path": path,
                "page_views": ga4_p["page_views"],
                "impressions": gsc_imp,
                "action": "add_internal_links",
                "target_page": path,
                "source_pages": source_pages,
                "description": f"{path} gets {ga4_p['page_views']} views but only {gsc_imp} search impr. Add internal links from index/blog.",
            })

    return {"tactics": tactics, "insights": insights}


def select_implementable_tactics(tactics: list, config: dict) -> list[dict]:
    """Filter tactics to those we can safely auto-implement, respecting limits."""
    exclude = set(config.get("exclude_patterns", []))
    max_meta = config.get("max_meta_updates", 5)
    max_low_hanging = config.get("max_low_hanging_fruit", 5)  # Separate quota so they don't crowd out declining
    max_competitor = config.get("max_competitor_meta_updates", 2)  # Competitor gap tactics
    max_links = config.get("max_internal_links_added", 3)
    max_onpage = config.get("max_onpage_fixes", 5)

    implementable = []
    meta_count = 0
    low_hanging_count = 0
    competitor_count = 0
    link_count = 0
    onpage_count = 0

    def score(t):
        pr = 0 if t["priority"] == "high" else 1
        imp = t.get("imp_drop") or t.get("impressions") or 0
        return (pr, -imp)

    for t in sorted(tactics, key=score):
        if any(pat in t.get("target_page", "") or pat in t.get("path", "") for pat in exclude):
            continue
        if t["action"] == "add_keyword_to_meta":
            if t.get("type") == "competitor_gap" and competitor_count < max_competitor:
                implementable.append(t)
                competitor_count += 1
            elif t.get("type") == "low_hanging_fruit" and low_hanging_count < max_low_hanging:
                implementable.append(t)
                low_hanging_count += 1
            elif t.get("type") not in ("low_hanging_fruit", "competitor_gap") and meta_count < max_meta:
                implementable.append(t)
                meta_count += 1
        elif t["action"] == "improve_meta_description" and meta_count < max_meta:
            implementable.append(t)
            meta_count += 1
        elif t["action"] == "fix_onpage_gaps" and onpage_count < max_onpage:
            implementable.append(t)
            onpage_count += 1
        elif t["action"] == "add_internal_links" and link_count < max_links:
            implementable.append(t)
            link_count += 1

    return implementable
