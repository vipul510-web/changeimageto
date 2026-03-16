"""
Implement SEO tactics by editing HTML files.
"""
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


def path_to_file(path: str, frontend_dir: str) -> Path:
    """Convert URL path to file path."""
    if path == "/":
        return PROJECT_ROOT / frontend_dir / "index.html"
    path = path.lstrip("/").rstrip("/")
    if path.endswith(".html"):
        return PROJECT_ROOT / frontend_dir / path
    # blog -> blog/index.html; bulk-image-resizer -> bulk-image-resizer.html
    index_path = PROJECT_ROOT / frontend_dir / path / "index.html"
    if index_path.exists():
        return index_path
    return PROJECT_ROOT / frontend_dir / (path + ".html")


def add_keyword_to_meta(html_path: Path, keyword: str) -> tuple[bool, str]:
    """
    Add keyword to meta keywords if not already present.
    Returns (success, message).
    """
    content = html_path.read_text(encoding="utf-8")
    if keyword.lower() in content.lower():
        return False, f"Keyword '{keyword}' already present"

    # Find meta name="keywords"
    meta_pattern = r'<meta\s+name=["\']keywords["\']\s+content=["\']([^"\']*)["\']'
    match = re.search(meta_pattern, content, re.I)
    if match:
        existing = match.group(1)
        if keyword in existing:
            return False, "Keyword already in meta"
        new_keywords = existing.rstrip(",") + ", " + keyword
        new_content = content.replace(match.group(0), f'<meta name="keywords" content="{new_keywords}"')
        html_path.write_text(new_content, encoding="utf-8")
        return True, f"Added '{keyword}' to meta keywords"
    return False, "No meta keywords tag found"


def improve_meta_description(html_path: Path, keyword: str) -> tuple[bool, str]:
    """
    Add keyword to meta description if it makes sense and isn't already there.
    """
    content = html_path.read_text(encoding="utf-8")
    if keyword.lower() in content.lower():
        return False, f"Keyword '{keyword}' already in description"

    meta_pattern = r'<meta\s+name=["\']description["\']\s+content=["\']([^"\']*)["\']'
    match = re.search(meta_pattern, content, re.I)
    if match:
        existing = match.group(1)
        if len(existing) > 155:
            return False, "Description already long"
        # Append keyword naturally if not too long
        addition = f" {keyword}." if not existing.endswith(".") else f" {keyword}"
        if len(existing) + len(addition) <= 160:
            new_desc = existing.rstrip(".") + addition
            new_content = content.replace(match.group(0), f'<meta name="description" content="{new_desc}"')
            html_path.write_text(new_content, encoding="utf-8")
            return True, f"Added '{keyword}' to meta description"
    return False, "No meta description or would exceed length"


def fix_onpage_gaps(html_path: Path, query: str, missing_in: list[str]) -> tuple[bool, str]:
    """
    Add missing keyword to title, meta description, or H1.
    Prioritizes: title > h1 > description (most SEO impact first).
    Returns (success, message).
    """
    content = html_path.read_text(encoding="utf-8")
    changes = []

    if "title" in missing_in:
        m = re.search(r"<title[^>]*>([^<]+)</title>", content, re.I)
        if m:
            existing = m.group(1).strip()
            if query.lower() not in existing.lower():
                # Append keyword if title isn't too long (keep under 60 chars)
                addition = f" | {query}" if len(existing) + len(query) + 3 <= 60 else ""
                if addition:
                    new_title = existing + addition
                    content = content.replace(m.group(0), f'<title>{new_title}</title>')
                    changes.append("title")

    if "h1" in missing_in and "h1" not in changes:
        m = re.search(r"<h1[^>]*>([^<]+)</h1>", content, re.I)
        if m:
            existing = m.group(1).strip()
            if query.lower() not in existing.lower():
                # Add keyword - e.g. "Free Tool" -> "Free Tool: {query}"
                new_h1 = f"{existing} – {query}" if len(existing) + len(query) + 3 <= 80 else existing
                if new_h1 != existing:
                    old_tag = m.group(0)
                    new_tag = old_tag.replace(m.group(1), new_h1)
                    content = content.replace(old_tag, new_tag)
                    changes.append("h1")

    if "description" in missing_in:
        m = re.search(r'<meta\s+name=["\']description["\']\s+content=["\']([^"\']*)["\']', content, re.I)
        if m:
            existing = m.group(1).strip()
            if query.lower() not in existing.lower() and len(existing) < 150:
                addition = f" {query}." if not existing.endswith(".") else f" {query}"
                if len(existing) + len(addition) <= 160:
                    new_desc = existing.rstrip(".") + addition
                    content = content.replace(m.group(0), f'<meta name="description" content="{new_desc}"')
                    changes.append("description")

    if changes:
        html_path.write_text(content, encoding="utf-8")
        return True, f"Added '{query}' to {', '.join(changes)}"
    return False, f"Could not add '{query}' (constraints or already present)"


# Human-readable link text for tool pages (path -> anchor text)
PAGE_LINK_TEXT = {
    "/": "ChangeImageTo.com",
    "/bulk-image-resizer.html": "Bulk Image Resizer",
    "/remove-background-from-image.html": "Remove Background",
    "/change-image-background.html": "Change Image Background",
    "/change-image-background-to-white.html": "White Background",
    "/change-image-background-to-black.html": "Black Background",
    "/change-image-background-to-blue.html": "Blue Background",
    "/blur-background.html": "Blur Background",
    "/remove-background-for-amazon.html": "Amazon Product Photos",
}


def _path_to_link_text(path: str) -> str:
    """Derive anchor text from path."""
    if path in PAGE_LINK_TEXT:
        return PAGE_LINK_TEXT[path]
    # Fallback: /bulk-image-resizer.html -> "Bulk Image Resizer"
    name = path.rstrip("/").replace("/", " ").replace(".html", "").replace("-", " ").strip()
    return name.title() if name else "Home"


def add_internal_link(html_path: Path, link_text: str, link_url: str, section: str = "main") -> tuple[bool, str]:
    """
    Add an internal link in a relevant section. Looks for a good place to add.
    """
    content = html_path.read_text(encoding="utf-8")
    # Check if we already have a link to this URL
    if f'href="{link_url}"' in content or f"href='{link_url}'" in content:
        return False, f"Link to {link_url} already exists"

    link_html = f'<a href="{link_url}">{link_text}</a>'
    # Try to add before footer or </main>
    insert_before = -1
    if "</main>" in content:
        insert_before = content.find("</main>")
    elif "<footer" in content:
        insert_before = content.find("<footer")
    if insert_before >= 0:
        snippet = f' <p class="seo-related">Related: {link_html}</p>\n  '
        new_content = content[:insert_before] + snippet + content[insert_before:]
        html_path.write_text(new_content, encoding="utf-8")
        return True, f"Added internal link to {link_url}"
    return False, "Could not find insertion point"


def implement_tactic(tactic: dict, frontend_dir: str) -> dict:
    """
    Implement a single tactic. Returns {success, message, file_changed}.
    """
    action = tactic.get("action")
    target = tactic.get("target_page") or tactic.get("path")
    if not target:
        return {"success": False, "message": "No target page", "file_changed": None}

    html_path = path_to_file(target, frontend_dir)
    if not html_path.exists():
        return {"success": False, "message": f"File not found: {html_path}", "file_changed": None}

    keyword = tactic.get("keyword") or tactic.get("query", "")
    if action == "add_keyword_to_meta":
        success, msg = add_keyword_to_meta(html_path, keyword)
        return {"success": success, "message": msg, "file_changed": str(html_path) if success else None}
    elif action == "improve_meta_description":
        success, msg = improve_meta_description(html_path, keyword)
        return {"success": success, "message": msg, "file_changed": str(html_path) if success else None}
    elif action == "fix_onpage_gaps":
        missing = tactic.get("missing_in", [])
        if not missing:
            return {"success": False, "message": "No missing fields", "file_changed": None}
        success, msg = fix_onpage_gaps(html_path, keyword, missing)
        return {"success": success, "message": msg, "file_changed": str(html_path) if success else None}
    elif action == "add_internal_links":
        link_url = target if target.startswith("/") else "/" + target.lstrip("/")
        link_text = _path_to_link_text(link_url)
        source_pages = tactic.get("source_pages", ["/", "/blog/"])
        added = []
        for src in source_pages:
            src_path = path_to_file(src, frontend_dir)
            if not src_path.exists():
                continue
            success, msg = add_internal_link(src_path, link_text, link_url)
            if success:
                added.append(str(src_path))
        if added:
            return {"success": True, "message": f"Added link to {link_url} on {len(added)} page(s)", "file_changed": added}
        return {"success": False, "message": "Link already exists on source pages or no insertion point", "file_changed": None}
    return {"success": False, "message": f"Unknown action: {action}", "file_changed": None}
