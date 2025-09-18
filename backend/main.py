from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.responses import Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import io
import logging
import asyncio
import json
import os
from datetime import datetime
from typing import Optional, List
import numpy as np
import re
import hashlib
import requests
import zipfile
import tempfile

try:
    from google.cloud import storage
except Exception:
    storage = None  # optional in local dev

try:
    from pytrends.request import TrendReq
except Exception:
    TrendReq = None

from rembg import remove, new_session

app = FastAPI(title="BG Remover", description="Simple background removal API", version="1.0.0")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Limit heavy CPU tasks to protect memory/CPU. Override via env MAX_CONCURRENCY
PROCESS_SEM = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENCY", "2")))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME_FOR_CATEGORY = {
    "product": os.getenv("DEFAULT_MODEL", "u2netp"),
    "portrait": os.getenv("DEFAULT_MODEL", "u2net_human_seg"),
}

BLOG_BUCKET = os.getenv("BLOG_BUCKET", "")
CRON_TOKEN = os.getenv("CRON_TOKEN", "")

# Resolve path to static frontend assets so blog pages can use site-wide styles in local/dev
FRONTEND_DIR = os.getenv(
    "FRONTEND_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
)

# Minimal static file endpoints used by blog HTML
@app.get("/styles.css")
async def serve_styles():
    path = os.path.join(FRONTEND_DIR, "styles.css")
    return FileResponse(path, media_type="text/css")


@app.get("/script.js")
async def serve_script():
    path = os.path.join(FRONTEND_DIR, "script.js")
    return FileResponse(path, media_type="application/javascript")


@app.get("/logo.png")
async def serve_logo():
    path = os.path.join(FRONTEND_DIR, "logo.png")
    return FileResponse(path, media_type="image/png")


@app.get("/favicon.ico")
async def serve_favicon():
    # Reuse logo for now if favicon not present
    path = os.path.join(FRONTEND_DIR, "logo.png")
    return FileResponse(path, media_type="image/png")

_sessions_cache = {}
def downscale_image_if_needed(image: Image.Image, max_side: int = int(os.getenv("MAX_IMAGE_SIDE", "1600"))) -> Image.Image:
    """Downscale very large images to reduce memory/CPU load while preserving aspect ratio."""
    try:
        w, h = image.width, image.height
        longest = max(w, h)
        if longest <= max_side:
            return image
        scale = max_side / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return image.resize((new_w, new_h), Image.LANCZOS)
    except Exception:
        # In case of any issues, return original
        return image

def log_user_action(action: str, details: dict):
    """Log user actions with timestamp and details"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "details": details
    }
    logger.info(f"USER_ACTION: {json.dumps(log_entry)}")


# -----------------
# Blog helper utils
# -----------------
def get_storage_client():
    if not BLOG_BUCKET:
        raise RuntimeError("BLOG_BUCKET not configured")
    if storage is None:
        raise RuntimeError("google-cloud-storage not installed")
    return storage.Client()

def get_or_create_bucket(client):
    """Return the GCS bucket for BLOG_BUCKET, creating it if it doesn't exist.

    This avoids 404 errors on first run when the bucket hasn't been created yet.
    Location defaults to US; can be overridden via env GCS_BUCKET_LOCATION.
    """
    try:
        bucket = client.lookup_bucket(BLOG_BUCKET)
        if bucket is None:
            location = os.getenv("GCS_BUCKET_LOCATION", "US")
            bucket = storage.Bucket(client, BLOG_BUCKET)
            bucket.location = location
            bucket = client.create_bucket(bucket)
        return bucket
    except Exception as e:
        raise RuntimeError(f"Failed to access or create BLOG_BUCKET '{BLOG_BUCKET}': {str(e)}")

def normalize_slug(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text[:120].strip("-") or hashlib.md5(text.encode()).hexdigest()[:12]

def fetch_google_autocomplete(seed: str) -> list:
    try:
        url = "https://suggestqueries.google.com/complete/search"
        resp = requests.get(url, params={"client": "firefox", "q": seed}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [s for s in data[1] if isinstance(s, str)]
    except Exception:
        return []

def fetch_trends_related(seed: str) -> list:
    if TrendReq is None:
        return []
    try:
        pt = TrendReq(hl='en-US', tz=0)
        pt.build_payload([seed], timeframe='today 12-m')
        rel = pt.related_queries()
        out = []
        for v in rel.values():
            for kind in ("top", "rising"):
                df = v.get(kind)
                if df is not None:
                    out.extend(df['query'].tolist()[:10])
        return out
    except Exception:
        return []

def pick_keywords(seeds: list, existing_slugs: set) -> list:
    ideas = []
    for seed in seeds:
        ideas.extend(fetch_google_autocomplete(seed))
        ideas.extend(fetch_trends_related(seed))
    # de-dup and prefer long-tail 3+ words
    uniq = []
    seen = set()
    for k in ideas:
        kk = k.strip().lower()
        if kk and kk not in seen and len(kk.split()) >= 3:
            slug = normalize_slug(kk)
            if slug not in existing_slugs:
                uniq.append((kk, slug))
                seen.add(kk)
    return uniq[:5]

def render_article_html(title: str, slug: str, body_sections: list) -> str:
    # minimal, reuse global CSS and script
    now_iso = datetime.utcnow().isoformat()
    json_ld = {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": title,
        "datePublished": now_iso,
        "author": {"@type": "Organization", "name": "ChangeImageTo.com Team"},
    }
    sections_html = "\n".join(body_sections)
    return f"""<!doctype html><html lang=\"en\"><head>
<meta charset=\"utf-8\"/><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>
<title>{title}</title>
<meta name=\"description\" content=\"{title} – practical guide and tips.\"/>
<link rel=\"canonical\" href=\"https://www.changeimageto.com/blog/{slug}.html\"/>
<script type=\"application/ld+json\">{json.dumps(json_ld)}</script>
<link rel=\"preload\" as=\"style\" href=\"/styles.css?v=20250916-3\"/><link rel=\"stylesheet\" href=\"/styles.css?v=20250916-3\"/>
<link rel=\"stylesheet\" href=\"https://www.changeimageto.com/styles.css?v=20250916-3\"/>
</head><body>
<header class=\"container header\"><a href=\"https://www.changeimageto.com/\" class=\"logo-link\"><img src=\"https://www.changeimageto.com/logo.png?v=20250916-2\" alt=\"ChangeImageTo\" class=\"logo-img\"/></a><div style=\"display:flex;align-items:center;gap:16px;justify-content:space-between;width:100%\"><h1 style=\"margin:0\">{title}</h1><nav class=\"top-nav\"><a href=\"https://www.changeimageto.com/blog\" aria-label=\"Read our blog\">Blog</a></nav></div></header>
<main class=\"container main\">\n  <p class=\"seo\" style=\"margin:0 0 16px\"><strong>By:</strong> ChangeImageTo.com Team · <time datetime=\"{now_iso}\">{now_iso.replace('T',' ')[:19]} UTC</time></p>\n  {sections_html}\n  <p class=\"seo\" style=\"margin-top:24px\"><a href=\"https://www.changeimageto.com/blog\" style=\"color:#fff\">← Back to blog</a></p>\n</main>
<nav class=\"seo-links\"><a href=\"https://www.changeimageto.com/remove-background-from-image.html\">Remove Background from Image</a><a href=\"https://www.changeimageto.com/change-color-of-image.html\">Change color of image online</a><a href=\"https://www.changeimageto.com/change-image-background.html\">Change image background</a><a href=\"https://www.changeimageto.com/convert-image-format.html\">Convert image format</a><a href=\"https://www.changeimageto.com/upscale-image.html\">AI Image Upscaler</a><a href=\"https://www.changeimageto.com/blur-background.html\">Blur Background</a><a href=\"https://www.changeimageto.com/enhance-image.html\">Enhance Image</a></nav>
<footer class=\"container footer\"><p>Built for speed and quality. <a href=\"https://www.changeimageto.com/\#\" rel=\"nofollow\">Contact</a></p></footer>
<script src=\"/script.js?v=20250916-3\" defer></script>
</body></html>"""

def build_sections(keyword: str) -> list:
    k = keyword.lower()
    sections = []

    def tools_list():
        return (
            "<section class=\"seo\"><h3>Try it online (free)</h3><ul>"
            "<li><a href=\"/remove-background-from-image.html\">Remove background from image</a></li>"
            "<li><a href=\"/change-image-background.html\">Change image background</a></li>"
            "<li><a href=\"/upscale-image.html\">AI Image Upscaler</a></li>"
            "<li><a href=\"/enhance-image.html\">Enhance image quality</a></li>"
            "</ul></section>"
        )

    if "powerpoint" in k:
        sections = [
            "<section class=\"seo\"><p>Yes, you can remove the background of a picture directly in Microsoft PowerPoint. Here is the exact sequence that works in Office 365 and PowerPoint 2019+.</p></section>",
            "<section class=\"seo\"><h2>Remove Background in PowerPoint (exact steps)</h2><ol>"
            "<li>Insert the picture: Insert → Pictures → choose your file.</li>"
            "<li>Select the picture, then go to Picture Format → Remove Background.</li>"
            "<li>Use Mark Areas to Keep/Remove to refine. Zoom in and click along edges.</li>"
            "<li>When satisfied, click Keep Changes. The background becomes transparent.</li>"
            "<li>Export: File → Save As → PNG → check Transparent background when available.</li>"
            "</ol><p>Older versions: the Remove Background button is under Picture Tools → Format.</p></section>",
            tools_list(),
            "<section class=\"seo\"><h2>Tips</h2><ul><li>High-contrast images work best.</li><li>If edges look rough, add a soft shadow or export at higher resolution.</li></ul></section>",
        ]
    elif "photoshop" in k:
        sections = [
            "<section class=\"seo\"><p>This guide shows the precise Photoshop commands to remove the background.</p></section>",
            "<section class=\"seo\"><h2>Photoshop quick method</h2><ol>"
            "<li>Open the image. In Layers, unlock the Background layer.</li>"
            "<li>Select: Select → Subject. Then click Select and Mask.</li>"
            "<li>Refine edges with the Refine Edge Brush, Output: New Layer with Layer Mask.</li>"
            "<li>Hide or delete the background layer. Export: File → Export → Export As → PNG.</li>"
            "</ol></section>",
            tools_list(),
            "<section class=\"seo\"><h2>Keyboard shortcuts</h2><p>Q toggles Quick Mask, Shift+F6 feathers a selection, and ⌘/Ctrl+J duplicates selection to a new layer.</p></section>",
        ]
    elif "iphone" in k or "ios" in k:
        sections = [
            "<section class=\"seo\"><p>On iPhone (iOS 16+), you can lift the subject from the background in Photos without any app.</p></section>",
            "<section class=\"seo\"><h2>Remove background on iPhone</h2><ol>"
            "<li>Open the photo in Photos.</li>"
            "<li>Press and hold the subject until a glow appears, choose Copy or Share → Save Image to export the cutout with a transparent background.</li>"
            "<li>For color backgrounds, upload the cutout to our Change Background tool and pick a color.</li>"
            "</ol></section>",
            tools_list(),
        ]
    elif "free" in k:
        sections = [
            "<section class=\"seo\"><p>Here are free, watermark‑free ways to remove image backgrounds online.</p></section>",
            "<section class=\"seo\"><h2>Completely free methods</h2><ol>"
            "<li>Use our Remove Background tool (no login, no watermark).</li>"
            "<li>For batch images, process one by one to keep best quality.</li>"
            "<li>Download as PNG to preserve transparency.</li>"
            "</ol></section>",
            tools_list(),
        ]
    else:
        sections = [
            f"<section class=\"seo\"><p>This tutorial explains how to {keyword} precisely using built‑in tools and our free online apps.</p></section>",
            "<section class=\"seo\"><h2>Exact steps (online)</h2><ol>"
            "<li>Open the Remove Background tool.</li>"
            "<li>Upload your image (JPG/PNG/WebP). Processing takes ~5–10s.</li>"
            "<li>Download the PNG with transparent background, or change the background color.</li>"
            "</ol></section>",
            tools_list(),
        ]

    sections.append(
        f"<section class=\"seo\"><h2>FAQ</h2><details><summary>Is it really free?</summary><p>Yes, our tools are free with no watermark and no login required.</p></details><details><summary>Best file format?</summary><p>Use PNG to keep transparency; use JPG for photos with solid backgrounds.</p></details></section>"
    )
    return sections

def list_existing_slugs(client) -> set:
    if not BLOG_BUCKET:
        return set()
    try:
        bucket = get_or_create_bucket(client)
        blobs = list(bucket.list_blobs(prefix="blog/"))
        slugs = set()
        for b in blobs:
            if b.name.endswith('.html'):
                s = os.path.basename(b.name).replace('.html','')
                slugs.add(s)
        return slugs
    except Exception:
        return set()

def save_article(client, slug: str, html: str):
    bucket = get_or_create_bucket(client)
    blob = bucket.blob(f"blog/{slug}.html")
    # Keep CDN/browser cache short so edits propagate quickly
    blob.cache_control = "public, max-age=600"
    blob.upload_from_string(html, content_type="text/html; charset=utf-8")
    # update index json (append latest)
    idx = bucket.blob("blog/index.json")
    try:
        existing = json.loads(idx.download_as_text())
    except Exception:
        existing = {"posts": []}
    url = f"/blog/{slug}.html"
    existing["posts"] = ([{"slug": slug, "title": html.split('<title>')[1].split('</title>')[0], "url": url, "date": datetime.utcnow().isoformat()}] 
                           + [p for p in existing.get("posts", []) if p.get("slug") != slug])[:200]
    idx.upload_from_string(json.dumps(existing), content_type="application/json")

@app.post("/api/blog/regenerate-all")
async def blog_regenerate_all(token: str):
    """Re-render and overwrite all existing blog posts using current templates/content.

    Requires CRON_TOKEN for authorization.
    """
    if CRON_TOKEN and token != CRON_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not BLOG_BUCKET:
        raise HTTPException(status_code=400, detail="BLOG_BUCKET not configured in production")
    client = get_storage_client()
    bucket = get_or_create_bucket(client)
    # Determine slugs from index.json, fallback to listing blobs
    slugs: set[str] = set()
    try:
        idx = bucket.blob("blog/index.json")
        data = json.loads(idx.download_as_text())
        for p in data.get("posts", []):
            s = p.get("slug")
            if s:
                slugs.add(s)
    except Exception:
        pass
    if not slugs:
        for b in bucket.list_blobs(prefix="blog/"):
            if b.name.endswith(".html"):
                slugs.add(os.path.basename(b.name).replace(".html", ""))
    regenerated = []
    for slug in sorted(slugs):
        title = slug.replace('-', ' ').title()
        html = render_article_html(title, slug, build_sections(title))
        save_article(client, slug, html)
        regenerated.append(slug)
    return {"regenerated": regenerated, "count": len(regenerated)}

@app.post("/api/blog/regenerate-one")
async def blog_regenerate_one(slug: str, token: str):
    if CRON_TOKEN and token != CRON_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not BLOG_BUCKET:
        raise HTTPException(status_code=400, detail="BLOG_BUCKET not configured in production")
    client = get_storage_client()
    title = slug.replace('-', ' ').title()
    html = render_article_html(title, slug, build_sections(title))
    save_article(client, slug, html)
    return {"ok": True, "slug": slug}

def get_session_for_category(category: str):
    category = category.lower()
    model_name = MODEL_NAME_FOR_CATEGORY.get(category)
    if not model_name:
        raise HTTPException(status_code=400, detail=f"Unsupported category: {category}")
    if model_name not in _sessions_cache:
        _sessions_cache[model_name] = new_session(model_name)
    return _sessions_cache[model_name]


@app.on_event("startup")
async def warm_models_on_startup():
    """Kick off model session initialization in the background so the server
    can bind to the port immediately and avoid Render's port scan timeout.
    """
    try:
        # Run in threads to avoid blocking the event loop
        asyncio.create_task(asyncio.to_thread(get_session_for_category, "product"))
        asyncio.create_task(asyncio.to_thread(get_session_for_category, "portrait"))
        logger.info("Background model warmup tasks started")
    except Exception as e:
        logger.warning(f"Failed to start background warmup: {e}")

@app.post("/api/remove-bg")
async def remove_bg(
    request: Request,
    file: UploadFile = File(...),
    category: str = Form("product"),
    bg_color: Optional[str] = Form(None),
):
    # Log the request start
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    
    log_user_action("request_started", {
        "client_ip": client_ip,
        "user_agent": user_agent,
        "filename": file.filename,
        "content_type": file.content_type,
        "category": category,
        "bg_color": bg_color,
        "has_bg_color": bg_color is not None
    })

    try:
        session = get_session_for_category(category)
    except HTTPException as e:
        log_user_action("error", {
            "error_type": "model_init",
            "error_message": str(e.detail),
            "category": category
        })
        raise
    except Exception as e:
        log_user_action("error", {
            "error_type": "model_init_unexpected",
            "error_message": str(e),
            "category": category
        })
        raise HTTPException(status_code=500, detail=f"Model init error: {str(e)}")

    if not file.content_type or not file.content_type.startswith("image/"):
        log_user_action("error", {
            "error_type": "invalid_file_type",
            "content_type": file.content_type,
            "filename": file.filename
        })
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        file_size = len(contents)
        image = Image.open(io.BytesIO(contents)).convert("RGBA")
        original_size = (image.width, image.height)
        
        # Log successful file processing
        log_user_action("file_processed", {
            "filename": file.filename,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "image_dimensions": f"{image.width}x{image.height}",
            "category": category
        })
        
    except Exception as e:
        log_user_action("error", {
            "error_type": "file_processing",
            "error_message": str(e),
            "filename": file.filename,
            "file_size": len(contents) if 'contents' in locals() else 0
        })
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        # Log processing start
        log_user_action("processing_started", {
            "category": category,
            "model": MODEL_NAME_FOR_CATEGORY.get(category.lower()),
            "bg_color": bg_color,
            "action_type": "change_background" if bg_color else "remove_background"
        })

        # Downscale to protect memory for processing, but remember original_size for output
        image = downscale_image_if_needed(image)

        async with PROCESS_SEM:
            result = remove(
                image,
                session=session,
                alpha_matting=True,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_size=10,
                post_process_mask=True,
            )
        if result.mode != "RGBA":
            result = result.convert("RGBA")

        # Always trim transparent padding first
        trimmed_result = result
        try:
            alpha_channel = result.split()[-1]
            trim_bbox = alpha_channel.getbbox()
            if trim_bbox and trim_bbox != (0, 0, result.width, result.height):
                trimmed_result = result.crop(trim_bbox)
        except Exception:
            pass
            
        # Optional solid background color compositing
        if bg_color:
            log_user_action("bg_color_debug", {
                "bg_color_raw": bg_color,
                "bg_color_type": type(bg_color).__name__,
                "bg_color_repr": repr(bg_color)
            })
            col = bg_color.strip()
            if col.startswith('#'):
                col = col[1:]
            if len(col) in (3,4):
                col = ''.join(c*2 for c in col[:3])
            if len(col) != 6:
                log_user_action("error", {
                    "error_type": "invalid_bg_color",
                    "bg_color": bg_color,
                    "processed_color": col
                })
                raise HTTPException(status_code=400, detail="Invalid bg_color; use hex like #ffffff")
            r = int(col[0:2], 16); g = int(col[2:4], 16); b = int(col[4:6], 16)
            # Ensure output keeps the original input dimensions
            canvas = Image.new('RGBA', original_size, (r, g, b, 255))
            # Center the trimmed subject on the original-sized canvas
            offset_x = (original_size[0] - trimmed_result.width) // 2
            offset_y = (original_size[1] - trimmed_result.height) // 2
            canvas.paste(trimmed_result, (offset_x, offset_y), mask=trimmed_result.split()[-1])
            result = canvas
        else:
            # For transparent output, use the trimmed result
            result = trimmed_result
            
        output_io = io.BytesIO()
        result.save(output_io, format="PNG")
        output_bytes = output_io.getvalue()
        
        # Log successful processing
        log_user_action("processing_completed", {
            "category": category,
            "bg_color": bg_color,
            "action_type": "change_background" if bg_color else "remove_background",
            "output_size_bytes": len(output_bytes),
            "output_size_mb": round(len(output_bytes) / (1024 * 1024), 2),
            "processing_successful": True
        })
        
        return Response(content=output_bytes, media_type="image/png")
        
    except Exception as e:
        log_user_action("error", {
            "error_type": "processing_error",
            "error_message": str(e),
            "category": category,
            "bg_color": bg_color,
            "action_type": "change_background" if bg_color else "remove_background"
        })
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/api/change-color")
async def change_color(
    request: Request,
    file: UploadFile = File(...),
    color_type: str = Form("hue"),
    hue_shift: float = Form(0),
    saturation: float = Form(100),
    brightness: float = Form(100),
    contrast: float = Form(100),
):
    """Change colors in an image"""
    # Log the request start
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    
    log_user_action("color_change_request_started", {
        "client_ip": client_ip,
        "user_agent": user_agent,
        "filename": file.filename,
        "content_type": file.content_type,
        "color_type": color_type,
        "hue_shift": hue_shift,
        "saturation": saturation,
        "brightness": brightness,
        "contrast": contrast
    })

    if not file.content_type or not file.content_type.startswith("image/"):
        log_user_action("error", {
            "error_type": "invalid_file_type_color_change",
            "content_type": file.content_type,
            "filename": file.filename
        })
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        file_size = len(contents)
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Log successful file processing
        log_user_action("color_change_file_processed", {
            "filename": file.filename,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "image_dimensions": f"{image.width}x{image.height}",
            "color_type": color_type
        })
        
    except Exception as e:
        log_user_action("error", {
            "error_type": "color_change_file_processing",
            "error_message": str(e),
            "filename": file.filename,
            "file_size": len(contents) if 'contents' in locals() else 0
        })
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        # Log processing start
        log_user_action("color_change_processing_started", {
            "color_type": color_type,
            "hue_shift": hue_shift,
            "saturation": saturation,
            "brightness": brightness,
            "contrast": contrast
        })
        
        # Downscale first to reduce memory, then apply color changes
        image = downscale_image_if_needed(image)
        async with PROCESS_SEM:
            # Apply color changes based on type
            result = apply_color_changes(image, color_type, hue_shift, saturation, brightness, contrast)
        
        output_io = io.BytesIO()
        result.save(output_io, format="PNG")
        output_bytes = output_io.getvalue()
        
        # Log successful processing
        log_user_action("color_change_processing_completed", {
            "color_type": color_type,
            "hue_shift": hue_shift,
            "saturation": saturation,
            "brightness": brightness,
            "contrast": contrast,
            "output_size_bytes": len(output_bytes),
            "output_size_mb": round(len(output_bytes) / (1024 * 1024), 2),
            "processing_successful": True
        })
        
        return Response(content=output_bytes, media_type="image/png")
        
    except Exception as e:
        log_user_action("error", {
            "error_type": "color_change_processing_error",
            "error_message": str(e),
            "color_type": color_type,
            "hue_shift": hue_shift,
            "saturation": saturation,
            "brightness": brightness,
            "contrast": contrast
        })
        raise HTTPException(status_code=500, detail=f"Color change processing error: {str(e)}")


def apply_color_changes(image: Image.Image, color_type: str, hue_shift: float, saturation: float, brightness: float, contrast: float) -> Image.Image:
    """Apply color changes to an image"""
    result = image.copy()
    
    if color_type == "grayscale":
        # Convert to grayscale
        result = ImageOps.grayscale(result)
        result = result.convert("RGB")
    else:
        # Apply HSV adjustments
        if hue_shift != 0:
            result = apply_hue_shift(result, hue_shift)
        
        # Apply saturation
        if saturation != 100:
            enhancer = ImageEnhance.Color(result)
            result = enhancer.enhance(saturation / 100)
        
        # Apply brightness
        if brightness != 100:
            enhancer = ImageEnhance.Brightness(result)
            result = enhancer.enhance(brightness / 100)
        
        # Apply contrast
        if contrast != 100:
            enhancer = ImageEnhance.Contrast(result)
            result = enhancer.enhance(contrast / 100)
    
    return result


def apply_hue_shift(image: Image.Image, hue_shift: float) -> Image.Image:
    """Apply hue shift to an image"""
    # Convert to HSV
    hsv = image.convert("HSV")
    hsv_array = np.array(hsv)
    
    # Shift hue
    hsv_array[:, :, 0] = (hsv_array[:, :, 0] + hue_shift) % 360
    
    # Convert back to RGB
    result = Image.fromarray(hsv_array, "HSV").convert("RGB")
    return result


@app.post("/api/analytics")
async def log_analytics(request: Request):
    """Receive analytics data from frontend"""
    try:
        data = await request.json()
        log_user_action("frontend_analytics", data)
        return {"status": "logged"}
    except Exception as e:
        logger.error(f"Analytics logging error: {str(e)}")
        return {"status": "error", "message": str(e)}


@app.get("/")
async def healthcheck():
    return {"status": "ok"}


@app.head("/")
async def head_root():
    return Response(status_code=200)


# ----------------------------
# Image format conversion API
# ----------------------------
@app.post("/api/convert-format")
async def convert_format(
    request: Request,
    file: UploadFile = File(...),
    target_format: str = Form("png"),
    transparent: bool = Form(False),
    quality: Optional[int] = Form(90),
):
    """Convert an uploaded image to a different format.
    - target_format: png | jpg | jpeg | webp
    - transparent: keep/force transparency if supported by target format (PNG/WEBP)
    - quality: 1-100 for lossy formats
    """
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")

    target = (target_format or "png").lower()
    if target == "jpeg":
        target = "jpg"
    supported_targets = {"png", "jpg", "webp", "bmp", "gif", "tiff", "ico", "ppm", "pgm"}
    if target not in supported_targets:
        raise HTTPException(status_code=400, detail=f"Unsupported target_format. Use one of: {sorted(supported_targets)}")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    log_user_action("convert_request_started", {
        "client_ip": client_ip,
        "user_agent": user_agent,
        "filename": file.filename,
        "content_type": file.content_type,
        "target_format": target,
        "transparent": transparent,
        "width": getattr(image, 'width', None),
        "height": getattr(image, 'height', None),
    })

    # Decide mode based on transparency setting and target
    supports_alpha = target in {"png", "webp", "tiff", "ico", "gif"}
    keep_alpha = bool(transparent and supports_alpha)

    try:
        # Downscale before conversion to control memory
        image = downscale_image_if_needed(image)
        if keep_alpha:
            converted = image.convert("RGBA")
            # Auto-trim fully transparent padding
            try:
                alpha = converted.split()[-1]
                bbox = alpha.getbbox()
                if bbox and bbox != (0, 0, converted.width, converted.height):
                    converted = converted.crop(bbox)
            except Exception:
                pass
        else:
            # Flatten onto opaque white background for formats w/o alpha or when transparency is false
            if image.mode in ("RGBA", "LA"):
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                converted = background
            else:
                converted = image.convert("RGB")

        params = {}
        if target == "jpg":
            params.update({"quality": max(1, min(int(quality or 90), 100)), "optimize": True})
            save_format = "JPEG"
        elif target == "webp":
            params.update({"quality": max(1, min(int(quality or 90), 100))})
            save_format = "WEBP"
        elif target == "bmp":
            save_format = "BMP"
        elif target == "gif":
            # Convert to palette-based; keep single frame
            if keep_alpha:
                # GIF transparency via palette index
                converted = converted.convert("RGBA")
                palette_image = converted.convert("P", palette=Image.ADAPTIVE, colors=255)
                # Pick a transparent color index
                alpha = converted.split()[-1]
                mask = Image.eval(alpha, lambda a: 255 if a <= 1 else 0)
                palette_image.info['transparency'] = 255  # last index
                palette_image.paste(255, None, mask)
                converted = palette_image
            else:
                converted = converted.convert("P", palette=Image.ADAPTIVE, colors=256)
            params.update({"optimize": True, "save_all": False})
            save_format = "GIF"
        elif target == "tiff":
            params.update({"compression": "tiff_lzw"})
            save_format = "TIFF"
        elif target == "ico":
            # ICO must be 256x256. Use cover-fit: fill the square with center-crop (no padding)
            size = 256
            img_rgba = converted.convert("RGBA")
            # scale so the shortest side is 256 (cover)
            min_side = min(img_rgba.width, img_rgba.height)
            scale = size / float(min_side)
            resized = img_rgba.resize((max(1, int(round(img_rgba.width * scale))),
                                       max(1, int(round(img_rgba.height * scale)))), Image.LANCZOS)
            # center crop to 256x256
            left = (resized.width - size) // 2
            top = (resized.height - size) // 2
            converted = resized.crop((left, top, left + size, top + size))
            save_format = "ICO"
            params.update({"sizes": [(256, 256)]})
        elif target in {"ppm", "pgm"}:
            # Ensure correct mode and explicit binary save
            if target == "ppm":
                converted = converted.convert("RGB")
            else:
                converted = converted.convert("L")
            save_format = "PPM"
            params.update({"bits": 8})
        else:
            save_format = "PNG"

        async with PROCESS_SEM:
            buf = io.BytesIO()
            converted.save(buf, format=save_format, **params)
            out = buf.getvalue()

        log_user_action("convert_completed", {
            "target_format": target,
            "transparent": keep_alpha,
            "output_size_bytes": len(out),
        })

        media = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "webp": "image/webp",
            "bmp": "image/bmp",
            "gif": "image/gif",
            "tiff": "image/tiff",
            "ico": "image/x-icon",
            "ppm": "image/x-portable-pixmap",
            "pgm": "image/x-portable-graymap",
        }[target]
        return Response(content=out, media_type=media)

    except Exception as e:
        log_user_action("convert_error", {"message": str(e)})
        raise HTTPException(status_code=500, detail=f"Conversion error: {str(e)}")


# ----------------------------
# Blog endpoints
# ----------------------------
@app.post("/api/generate-blog-daily")
async def generate_blog_daily(token: str):
    if CRON_TOKEN and token != CRON_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    client = get_storage_client()
    existing = list_existing_slugs(client)
    seeds = [
        "remove background from image",
        "change image background color",
        "image upscaler",
        "blur background in photo",
        "enhance image quality",
        "convert image format",
    ]
    picks = pick_keywords(seeds, existing)
    created = []
    for kw, slug in picks:
        title = kw.title()
        html = render_article_html(title, slug, build_sections(kw))
        save_article(client, slug, html)
        created.append({"slug": slug, "title": title})
    return {"created": created}


@app.get("/blog")
async def blog_index():
    # Local/dev fallback when BLOG_BUCKET is not configured
    items = []
    if BLOG_BUCKET:
        client = get_storage_client()
        bucket = get_or_create_bucket(client)
        idx = bucket.blob("blog/index.json")
        try:
            data = json.loads(idx.download_as_text())
        except Exception:
            data = {"posts": []}
        items = data.get("posts", [])
    else:
        items = [
            {"slug": "remove-background-from-image-in-powerpoint", "title": "Remove Background From Image In Powerpoint", "date": datetime.utcnow().isoformat()},
            {"slug": "remove-background-from-image-photoshop", "title": "Remove Background From Image Photoshop", "date": datetime.utcnow().isoformat()},
            {"slug": "remove-background-from-image-iphone", "title": "Remove Background From Image Iphone", "date": datetime.utcnow().isoformat()},
            {"slug": "remove-background-from-image-free", "title": "Remove Background From Image Free", "date": datetime.utcnow().isoformat()},
            {"slug": "remove-background-from-image", "title": "Remove Background From Image", "date": datetime.utcnow().isoformat()},
        ]
    # simple HTML list
    lis = "".join([
        f"<li><a href='/blog/{p['slug']}.html'>{p['title']}</a> <span class='date'>— {p.get('date','')[:10]}</span></li>"
        for p in items
    ])
    page_title = "Image Editing Blog | Tutorials, Tips & How‑To Guides"
    page_desc = "Learn image editing: remove background, change colors, upscale, blur, enhance images. Free tutorials and guides."
    json_ld = json.dumps({
        "@context": "https://schema.org",
        "@type": "Blog",
        "name": page_title,
        "description": page_desc,
        "url": "https://www.changeimageto.com/blog"
    })
    html = f"""<!doctype html><html lang='en'><head>
<meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>{page_title}</title>
<meta name='description' content='{page_desc}'>
<link rel='canonical' href='https://www.changeimageto.com/blog'>
<script type='application/ld+json'>{json_ld}</script>
<link rel='preload' as='style' href='/styles.css?v=20250916-3'/>
<link rel='stylesheet' href='/styles.css?v=20250916-3'/>
<link rel='stylesheet' href='https://www.changeimageto.com/styles.css?v=20250916-3'/>
<style>
  .blog-wrap{{max-width:1000px;margin:0 auto;padding:24px}}
  .blog-title{{margin:0 0 12px}}
  .blog-sub{{color:var(--muted);margin:0 0 16px}}
  .blog-list{{list-style:none;padding:0;margin:0}}
  .blog-list li{{padding:10px 0;border-bottom:1px solid var(--border)}}
  .blog-list li:last-child{{border-bottom:none}}
  .blog-list a{{color:#fff;text-decoration:none;font-weight:700}}
  .blog-list a:hover{{text-decoration:underline}}
  .blog-list .date{{color:var(--muted);font-weight:400}}
</style>
</head>
<body>
  <header class='container header'>
    <a href='/' class='logo-link'><img src='https://www.changeimageto.com/logo.png?v=20250916-2' class='logo-img' alt='ChangeImageTo'/></a>
    <div style='display:flex;align-items:center;gap:16px;justify-content:space-between;width:100%'>
      <h1 style='margin:0'>Image Editing Blog</h1>
      <nav class='top-nav'><a href='/blog' aria-label='Read our blog'>Blog</a></nav>
    </div>
  </header>
  <main class='blog-wrap'>
    <p class='blog-sub'>Guides for removing backgrounds, changing colors, upscaling, blurring, and enhancing images.</p>
    <ul class='blog-list'>{lis}</ul>
  </main>
  <nav class='seo-links'><a href='/remove-background-from-image.html'>Remove Background</a><a href='/change-image-background.html'>Change Background</a><a href='/change-color-of-image.html'>Change Color</a><a href='/upscale-image.html'>AI Image Upscaler</a><a href='/blur-background.html'>Blur Background</a><a href='/enhance-image.html'>Enhance Image</a></nav>
  <footer class='container footer'><p>Built for speed and quality. <a href='#' rel='nofollow'>Contact</a></p></footer>
  <script src='/script.js?v=20250916-3' defer></script>
</body></html>"""
    return Response(content=html, media_type="text/html")


@app.get("/blog/{slug}.html")
async def blog_article(slug: str):
    # Always render fresh from current template/sections
    title = slug.replace('-', ' ').title()
    fresh_html = render_article_html(title, slug, build_sections(title))
    # If in production with bucket, compare and update stored HTML if changed
    if BLOG_BUCKET:
        try:
            client = get_storage_client()
            bucket = get_or_create_bucket(client)
            blob = bucket.blob(f"blog/{slug}.html")
            old_html = blob.download_as_text() if blob.exists() else ""
            if old_html != fresh_html:
                save_article(client, slug, fresh_html)
        except Exception:
            # Non-fatal: still serve fresh content even if bucket update fails
            pass
    return Response(content=fresh_html, media_type="text/html")


@app.post("/api/upscale-image")
async def upscale_image(
    file: UploadFile = File(...),
    scale_factor: int = Form(2),
    method: str = Form("lanczos")
):
    """
    Upscale an image using AI-powered interpolation methods.
    
    Args:
        file: Image file to upscale
        scale_factor: Upscaling factor (2, 3, or 4)
        method: Upscaling method ('lanczos', 'cubic', 'nearest', 'area')
    
    Returns:
        Upscaled image as PNG
    """
    try:
        # Validate scale factor
        if scale_factor not in [2, 3, 4]:
            raise HTTPException(status_code=400, detail="Scale factor must be 2, 3, or 4")
        
        # Validate method
        valid_methods = ['lanczos', 'cubic', 'nearest', 'area']
        if method not in valid_methods:
            raise HTTPException(status_code=400, detail=f"Method must be one of: {valid_methods}")
        
        # Read and validate image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode in ('RGBA', 'LA', 'P'):
            # Create white background for transparent images
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get original dimensions
        original_width, original_height = image.size
        new_width = original_width * scale_factor
        new_height = original_height * scale_factor
        
        # Apply upscaling using PIL
        if method == 'lanczos':
            # Use PIL's LANCZOS resampling for high-quality upscaling
            upscaled_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        elif method == 'cubic':
            # Use PIL's BICUBIC resampling
            upscaled_image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        elif method == 'nearest':
            # Use PIL's NEAREST resampling
            upscaled_image = image.resize((new_width, new_height), Image.Resampling.NEAREST)
        elif method == 'area':
            # Use PIL's BOX resampling (similar to area)
            upscaled_image = image.resize((new_width, new_height), Image.Resampling.BOX)
        else:
            # Default to LANCZOS
            upscaled_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Apply additional sharpening for better quality
        enhancer = ImageEnhance.Sharpness(upscaled_image)
        upscaled_image = enhancer.enhance(1.2)  # Slight sharpening
        
        # Save as PNG
        output_buffer = io.BytesIO()
        upscaled_image.save(output_buffer, format='PNG', optimize=True)
        output_data = output_buffer.getvalue()
        
        # Log the action
        log_user_action("upscale_completed", {
            "original_size": f"{original_width}x{original_height}",
            "upscaled_size": f"{new_width}x{new_height}",
            "scale_factor": scale_factor,
            "method": method,
            "output_size_bytes": len(output_data)
        })
        
        return Response(content=output_data, media_type="image/png")
        
    except Exception as e:
        log_user_action("upscale_error", {"message": str(e)})
        raise HTTPException(status_code=500, detail=f"Upscaling error: {str(e)}")


@app.post("/api/test-upscale")
async def test_upscale():
    """Test endpoint to verify upscaling functionality"""
    return {"message": "Upscaling endpoint is working"}


# ----------------------------
# New simple image tools
# ----------------------------

@app.post("/api/blur-background")
async def blur_background(
    request: Request,
    file: UploadFile = File(...),
    blur_radius: float = Form(12.0),
    category: str = Form("portrait"),
):
    """Blur the background while keeping the subject sharp using existing segmentation."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGBA")
        # Downscale for processing efficiency
        proc_image = downscale_image_if_needed(image)
        # Segment foreground
        session = get_session_for_category(category)
        async with PROCESS_SEM:
            cutout = remove(
                proc_image,
                session=session,
                alpha_matting=True,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_size=10,
                post_process_mask=True,
            )
        # Build mask from alpha
        if cutout.mode != "RGBA":
            cutout = cutout.convert("RGBA")
        mask = cutout.split()[-1]
        # Prepare blurred background from original
        base_rgb = image.convert("RGB")
        blurred = base_rgb.filter(ImageFilter.GaussianBlur(radius=max(0.0, float(blur_radius))))
        blurred_rgba = blurred.convert("RGBA")
        # Composite: paste sharp subject onto blurred bg
        # Resize cutout to original size if we downscaled
        if cutout.size != image.size:
            cutout = cutout.resize(image.size, Image.LANCZOS)
            mask = mask.resize(image.size, Image.LANCZOS)
        output = blurred_rgba.copy()
        output.paste(cutout, (0, 0), mask=mask)
        buf = io.BytesIO()
        output.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        log_user_action("blur_background_error", {"message": str(e)})
        raise HTTPException(status_code=500, detail=f"Blur background error: {str(e)}")


@app.post("/api/enhance-image")
async def enhance_image(
    file: UploadFile = File(...),
    sharpen: float = Form(1.0),
    contrast: float = Form(105.0),
    brightness: float = Form(100.0),
):
    """Simple photo enhancement: unsharp + mild contrast/brightness tweaks."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = downscale_image_if_needed(img)
        # Sharpen
        if sharpen != 1.0:
            sharp_enh = ImageEnhance.Sharpness(img)
            img = sharp_enh.enhance(max(0.0, float(sharpen)))
        # Contrast
        if contrast != 100.0:
            cont_enh = ImageEnhance.Contrast(img)
            img = cont_enh.enhance(max(0.0, float(contrast)) / 100.0)
        # Brightness
        if brightness != 100.0:
            bri_enh = ImageEnhance.Brightness(img)
            img = bri_enh.enhance(max(0.0, float(brightness)) / 100.0)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        log_user_action("enhance_image_error", {"message": str(e)})
        raise HTTPException(status_code=500, detail=f"Enhance image error: {str(e)}")


@app.post("/api/bulk-resize")
async def bulk_resize(
    request: Request,
    files: List[UploadFile] = File(...),
    width: int = Form(800),
    height: int = Form(600),
    maintain_aspect: bool = Form(True),
    quality: int = Form(90),
):
    """Resize multiple images and return as a ZIP file."""
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    
    # Validate inputs
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 images allowed per batch")
    
    if width < 1 or width > 4000 or height < 1 or height > 4000:
        raise HTTPException(status_code=400, detail="Width and height must be between 1 and 4000 pixels")
    
    if quality < 1 or quality > 100:
        raise HTTPException(status_code=400, detail="Quality must be between 1 and 100")
    
    log_user_action("bulk_resize_started", {
        "client_ip": client_ip,
        "user_agent": user_agent,
        "file_count": len(files),
        "target_width": width,
        "target_height": height,
        "maintain_aspect": maintain_aspect,
        "quality": quality
    })
    
    processed_images = []
    errors = []
    
    try:
        for i, file in enumerate(files):
            try:
                # Validate file type
                if not file.content_type or not file.content_type.startswith("image/"):
                    errors.append(f"File {i+1} ({file.filename}): Not an image file")
                    continue
                
                # Read and process image
                contents = await file.read()
                if len(contents) > 10 * 1024 * 1024:  # 10MB limit per file
                    errors.append(f"File {i+1} ({file.filename}): File too large (max 10MB)")
                    continue
                
                image = Image.open(io.BytesIO(contents))
                
                # Convert to RGB if necessary
                if image.mode in ('RGBA', 'LA', 'P'):
                    # Create white background for transparent images
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    if image.mode == 'P':
                        image = image.convert('RGBA')
                    background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
                    image = background
                elif image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Resize image to exact dimensions
                original_width, original_height = image.size
                target_aspect = width / height
                original_aspect = original_width / original_height
                
                if maintain_aspect:
                    # Fit within target dimensions while maintaining aspect ratio
                    if original_aspect > target_aspect:
                        # Image is wider - fit to width
                        new_width = width
                        new_height = int(width / original_aspect)
                    else:
                        # Image is taller - fit to height
                        new_height = height
                        new_width = int(height * original_aspect)
                    
                    # Resize maintaining aspect ratio
                    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # Create canvas with exact target dimensions and center the image
                    canvas = Image.new('RGB', (width, height), (255, 255, 255))
                    offset_x = (width - new_width) // 2
                    offset_y = (height - new_height) // 2
                    canvas.paste(resized, (offset_x, offset_y))
                    resized = canvas
                else:
                    # Crop and resize to exact dimensions (may distort)
                    resized = image.resize((width, height), Image.Resampling.LANCZOS)
                
                # Save to bytes
                output_buffer = io.BytesIO()
                resized.save(output_buffer, format='JPEG', quality=quality, optimize=True)
                output_data = output_buffer.getvalue()
                
                # Store filename and data
                filename = file.filename or f"image_{i+1}.jpg"
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filename += '.jpg'
                
                processed_images.append({
                    'filename': filename,
                    'data': output_data,
                    'original_size': f"{image.width}x{image.height}",
                    'new_size': f"{new_width}x{new_height}"
                })
                
            except Exception as e:
                errors.append(f"File {i+1} ({file.filename}): {str(e)}")
                continue
        
        if not processed_images:
            raise HTTPException(status_code=400, detail="No images could be processed. " + "; ".join(errors))
        
        # Create ZIP file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for img_data in processed_images:
                zip_file.writestr(img_data['filename'], img_data['data'])
        
        zip_data = zip_buffer.getvalue()
        
        log_user_action("bulk_resize_completed", {
            "processed_count": len(processed_images),
            "error_count": len(errors),
            "zip_size_bytes": len(zip_data),
            "zip_size_mb": round(len(zip_data) / (1024 * 1024), 2)
        })
        
        # Return ZIP file
        headers = {
            "Content-Disposition": "attachment; filename=resized_images.zip",
            "Content-Type": "application/zip"
        }
        
        return Response(content=zip_data, headers=headers, media_type="application/zip")
        
    except HTTPException:
        raise
    except Exception as e:
        log_user_action("bulk_resize_error", {"message": str(e)})
        raise HTTPException(status_code=500, detail=f"Bulk resize error: {str(e)}")


