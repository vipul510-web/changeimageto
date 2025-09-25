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
import cv2
from skimage import measure, filters, exposure
from skimage.metrics import structural_similarity as ssim
import sqlite3
from enum import Enum

try:
    from google.cloud import storage
except Exception:
    storage = None  # optional in local dev

try:
    from pytrends.request import TrendReq
except Exception:
    TrendReq = None

try:
    import pytesseract
except Exception:
    pytesseract = None

from rembg import remove, new_session
try:
    import onnxruntime as ort
except Exception:
    ort = None

app = FastAPI(title="BG Remover", description="Simple background removal API", version="1.0.0")

@app.get("/")
async def root():
    """Root endpoint with links to main features"""
    return {
        "message": "Image Processing API",
        "version": "1.0.0",
        "endpoints": {
            "blog_cms": "/blog-admin",
            "blog_index": "/blog", 
            "image_tools": "/api/",
            "docs": "/docs"
        }
    }

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

# Background warmup for LaMa to reduce first-hit latency on Remove People
@app.on_event("startup")
async def _warmup_lama_background():
    try:
        if os.getenv("MODEL_WARMUP", "true").lower() not in ("1","true","yes"):
            return
        # Warm in a thread to avoid blocking startup
        loop = asyncio.get_event_loop()
        def init_models():
            try:
                if os.getenv("USE_LAMA", "true").lower() in ("1","true","yes"):
                    _ = _get_lama_session()
            except Exception as e:
                logger.warning(f"Warmup failed: {e}")
        loop.run_in_executor(None, init_models)
    except Exception as e:
        logger.warning(f"Warmup scheduling failed: {e}")


# -----------------------------
# Inpainting helpers (exemplar + blending)
# -----------------------------
def _mask_center(binary_mask: np.ndarray) -> tuple:
    try:
        m = cv2.moments(binary_mask, binaryImage=True)
        if m["m00"] > 0:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"]) 
            return (cx, cy)
    except Exception:
        pass
    h, w = binary_mask.shape[:2]
    return (w // 2, h // 2)

def exemplar_inpaint_patchmatch(bgr_image: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    """Try exemplar-based inpainting using OpenCV contrib xphoto Shift-Map.
    If unavailable, returns original image unchanged.
    """
    try:
        if hasattr(cv2, 'xphoto') and hasattr(cv2.xphoto, 'inpaint'):
            dst = bgr_image.copy()
            cv2.xphoto.inpaint(bgr_image, binary_mask, dst, cv2.xphoto.INPAINT_SHIFTMAP)
            return dst
    except Exception as e:
        logger.warning(f"PatchMatch inpaint failed: {e}")
    return bgr_image

# -----------------------------
# LaMa ONNX integration (optional)
# -----------------------------
_LAMA_SESSION = None
_LAMA_PROVIDERS = None

def _maybe_download_lama(model_path: str) -> None:
    try:
        url = os.getenv("LAMA_ONNX_URL", "").strip()
        if not url or os.path.exists(model_path):
            return
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        import requests
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(model_path, "wb") as f:
            f.write(r.content)
    except Exception as e:
        logger.warning(f"LaMa ONNX download skipped/failed: {e}")

def _get_lama_session():
    global _LAMA_SESSION, _LAMA_PROVIDERS
    if _LAMA_SESSION is not None:
        return _LAMA_SESSION
    model_path = os.getenv("LAMA_ONNX_PATH", "").strip()
    if not model_path or ort is None:
        return None
    if not os.path.exists(model_path):
        _maybe_download_lama(model_path)
    if not os.path.exists(model_path):
        return None
    try:
        providers = ["CPUExecutionProvider"]
        _LAMA_PROVIDERS = providers
        _LAMA_SESSION = ort.InferenceSession(model_path, providers=providers)
        logger.info(f"LaMa ONNX loaded with providers: {_LAMA_SESSION.get_providers()}")
        return _LAMA_SESSION
    except Exception as e:
        logger.warning(f"Failed to load LaMa ONNX: {e}")
        return None

def lama_inpaint_onnx(bgr_image: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    """Attempt LaMa ONNX inference. Returns original on failure."""
    sess = _get_lama_session()
    if sess is None:
        return bgr_image

# -----------------------------
# LaMa PyTorch (lama-cleaner) integration (lazy)
# -----------------------------
def _get_lama_manager():
    global _LAMA_MANAGER
    if _LAMA_MANAGER is not None:
        return _LAMA_MANAGER
    if ModelManager is None:
        return None
    try:
        # Minimal config for CPU
        device = "cpu"
        sd = os.getenv("LAMA_SD", "lama")
        # Lazy download on first init handled by ModelManager
        _LAMA_MANAGER = ModelManager(name=sd, device=device)
        logger.info("LaMa PyTorch (lama-cleaner) initialized")
        return _LAMA_MANAGER
    except Exception as e:
        logger.warning(f"Failed to init LaMa (lama-cleaner): {e}")
        return None

def lama_inpaint_torch(bgr_image: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    mgr = _get_lama_manager()
    if mgr is None:
        return bgr_image
    try:
        img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mask = (binary_mask > 0).astype(np.uint8) * 255
        # Use resize strategy to control memory/time
        res = mgr.inpaint(img, mask, hd_strategy=HDStrategy.ORIGINAL)
        return cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.warning(f"LaMa torch inpaint failed, fallback: {e}")
        return bgr_image
    try:
        img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mask = (binary_mask > 0).astype(np.uint8) * 255
        h, w = img.shape[:2]
        # LaMa prefers multiples of 8
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h or pad_w:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        inp = img.astype(np.float32) / 255.0
        m = (mask.astype(np.float32) / 255.0)
        # BCHW
        inp = np.transpose(inp, (2, 0, 1))[None, ...]
        m = m[None, None, ...]
        # Find input names heuristically
        inputs = {sess.get_inputs()[0].name: inp, sess.get_inputs()[1].name: m}
        out = sess.run(None, inputs)[0]
        # CHW -> HWC
        out = np.squeeze(out, axis=0)
        out = np.transpose(out, (1, 2, 0))
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        if pad_h or pad_w:
            out = out[:h, :w]
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.warning(f"LaMa ONNX inference failed, falling back: {e}")
        return bgr_image

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
<style>
  /* Force readable white text on blog articles */
  body, .main, main.container.main, .seo, .seo p, .seo li, .seo h2, .seo h3, .seo details, .seo summary {{ color: #ffffff; }}
  .seo a {{ color: #9ccfff; }}
  .seo a:hover {{ text-decoration: underline; }}
  .seo-links a {{ color: #ffffff; }}
  .header h1 {{ color: #ffffff; }}
</style>
</head><body>
<header class=\"container header\"><a href=\"https://www.changeimageto.com/\" class=\"logo-link\"><img src=\"https://www.changeimageto.com/logo.png?v=20250916-2\" alt=\"ChangeImageTo\" class=\"logo-img\"/></a><div style=\"display:flex;align-items:center;gap:16px;justify-content:space-between;width:100%\"><h1 style=\"margin:0\">{title}</h1><nav class=\"top-nav\"><a href=\"https://www.changeimageto.com/blog\" aria-label=\"Read our blog\">Blog</a></nav></div></header>
<main class=\"container main\">\n  <p class=\"seo\" style=\"margin:0 0 16px\"><strong>By:</strong> ChangeImageTo.com Team · <time datetime=\"{now_iso}\">{now_iso.replace('T',' ')[:19]} UTC</time></p>\n  {sections_html}\n  <p class=\"seo\" style=\"margin-top:24px\"><a href=\"https://www.changeimageto.com/blog\" style=\"color:#fff\">← Back to blog</a></p>\n</main>
<nav class=\"seo-links\"><a href=\"https://www.changeimageto.com/remove-background-from-image.html\">Remove Background from Image</a><a href=\"https://www.changeimageto.com/change-color-of-image.html\">Change color of image online</a><a href=\"https://www.changeimageto.com/change-image-background.html\">Change image background</a><a href=\"https://www.changeimageto.com/convert-image-format.html\">Convert image format</a><a href=\"https://www.changeimageto.com/upscale-image.html\">AI Image Upscaler</a><a href=\"https://www.changeimageto.com/blur-background.html\">Blur Background</a><a href=\"https://www.changeimageto.com/enhance-image.html\">Enhance Image</a></nav>
<footer class=\"container footer\"><p>Built for speed and quality. <a href=\"https://www.changeimageto.com/#\" rel=\"nofollow\">Contact</a></p></footer>
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
    elif "android" in k:
        sections = [
            "<section class=\"seo\"><p>Removing backgrounds from images on Android devices is easier than you might think. Here are the best methods using built-in Android features and popular apps.</p></section>",
            "<section class=\"seo\"><h2>Method 1: Google Photos (Built-in)</h2><ol>"
            "<li>Open Google Photos app on your Android device.</li>"
            "<li>Select the image you want to edit.</li>"
            "<li>Tap the \"Edit\" button (pencil icon).</li>"
            "<li>Scroll down and tap \"Magic Eraser\" or \"Background Blur\".</li>"
            "<li>Use your finger to highlight the background areas you want to remove.</li>"
            "<li>Tap \"Done\" and save the edited image.</li>"
            "</ol><p>Note: Magic Eraser is available on Google Pixel devices and newer Android phones with Google Photos.</p></section>",
            "<section class=\"seo\"><h2>Method 2: Samsung Gallery Editor</h2><ol>"
            "<li>Open Samsung Gallery app on your Samsung device.</li>"
            "<li>Select the image and tap \"Edit\".</li>"
            "<li>Look for \"Object Eraser\" or \"Background Eraser\" in the tools.</li>"
            "<li>Draw over the background areas you want to remove.</li>"
            "<li>Tap \"Apply\" and save your changes.</li>"
            "</ol><p>This feature is available on Samsung Galaxy devices with One UI 3.0 or later.</p></section>",
            "<section class=\"seo\"><h2>Method 3: Third-party Apps</h2><ol>"
            "<li><strong>Background Eraser:</strong> Download from Google Play Store, upload image, use auto-detection.</li>"
            "<li><strong>Remove.bg:</strong> Install the app, take or select photo, automatic background removal.</li>"
            "<li><strong>PhotoRoom:</strong> Professional-grade background removal with manual editing tools.</li>"
            "</ol></section>",
            "<section class=\"seo\"><h2>Method 4: Our Free Online Tool (Best Results)</h2><ol>"
            "<li>Open your Android browser and go to our <a href=\"/remove-background-from-image.html\">Remove Background</a> tool.</li>"
            "<li>Upload your image from your phone's gallery.</li>"
            "<li>Wait for automatic processing (5-10 seconds).</li>"
            "<li>Download the result with transparent background.</li>"
            "<li>Save to your Android device's gallery.</li>"
            "</ol><p>This method works on any Android device and produces professional-quality results.</p></section>",
            tools_list(),
            "<section class=\"seo\"><h2>Tips for Android Background Removal</h2><ul>"
            "<li>Use high-resolution images (at least 1080p) for better results.</li>"
            "<li>Ensure good lighting when taking photos for background removal.</li>"
            "<li>Images with clear subject-background contrast work best.</li>"
            "<li>Save results as PNG format to preserve transparency.</li>"
            "<li>Use landscape mode for better editing precision on tablets.</li>"
            "</ul></section>",
            "<section class=\"seo\"><h2>FAQ</h2>"
            "<details><summary>Which Android version supports background removal?</summary><p>Most modern Android devices (Android 8.0+) support basic editing features. Advanced AI background removal requires Android 10+ or specific manufacturer apps.</p></details>"
            "<details><summary>Is it really free on Android?</summary><p>Yes, our online tools are completely free on Android with no watermarks or login required.</p></details>"
            "<details><summary>Best Android app for background removal?</summary><p>Google Photos (Pixel devices), Samsung Gallery (Samsung devices), or our free online tool for any Android device.</p></details>"
            "<details><summary>Can I remove backgrounds from videos on Android?</summary><p>Yes, some apps like CapCut and InShot support video background removal on Android.</p></details>"
            "</section>",
        ]
    elif "google slides" in k or "google-slides" in k or "google" in k and "slides" in k:
        sections = [
            "<section class=\"seo\"><p>This guide shows the exact steps to remove a picture background in Google Slides without third‑party add‑ons. You can mask, crop, and make backgrounds transparent using built‑in tools and a quick detour via Google Drawings.</p></section>",
            "<section class=\"seo\"><h2>Method 1 — Make white background transparent</h2><ol>"
            "<li>Insert the image: Insert → Image → Upload from computer.</li>"
            "<li>Select the image → Format options (right panel) → Adjustments.</li>"
            "<li>Increase Transparency until the white/solid background fades. Fine‑tune Brightness/Contrast to keep the subject visible.</li>"
            "<li>Optional: Add a colored shape behind the image (Insert → Shape) for a new background.</li>"
            "</ol><p>Works best when the background is plain white or a single color.</p></section>",
            "<section class=\"seo\"><h2>Method 2 — Background removal via Google Drawings (free)</h2><ol>"
            "<li>Right‑click the image → Open with → Google Drawings.</li>"
            "<li>In Drawings, use the <em>Format → Format options → Adjustments</em> slider and the <em>Image → Recolor</em> if needed to isolate the subject.</li>"
            "<li>For tighter edges, use the <em>Crop</em> tool with <em>Mask</em> (dropdown next to the crop icon) and choose a shape that matches your subject. Drag the control handles to refine.</li>"
            "<li>File → Download → PNG. This preserves transparency.</li>"
            "<li>Back in Slides, Insert → Image → Upload → choose the PNG you downloaded.</li>"
            "</ol><p>Tip: PNG is required to keep transparent backgrounds.</p></section>",
            tools_list(),
            "<section class=\"seo\"><h2>Method 3 — Quick online cutout (best edges)</h2><ol>"
            "<li>Open our <a href=\"/remove-background-from-image.html\">Remove Background</a> tool.</li>"
            "<li>Upload your picture → download the PNG with transparent background.</li>"
            "<li>Insert the PNG into Google Slides and place a new background (Insert → Image → behind text).</li>"
            "</ol></section>",
            "<section class=\"seo\"><h2>Tips for Slides</h2><ul>"
            "<li>Use high‑resolution images (at least 1600×1200) to avoid pixelation on projectors.</li>"
            "<li>Add a soft shadow (Format options → Drop shadow) to make the cutout blend naturally.</li>"
            "<li>Lock background elements by setting them as the slide background image.</li>"
            "</ul></section>",
        ]
    elif "canva" in k:
        sections = [
            "<section class=\"seo\"><p>Canva has a built-in background remover that works well for simple images. Here's how to use it effectively.</p></section>",
            "<section class=\"seo\"><h2>Remove background in Canva (step-by-step)</h2><ol>"
            "<li>Upload your image to Canva or drag it from the Photos tab.</li>"
            "<li>Select the image and click <strong>Edit image</strong> (or Effects).</li>"
            "<li>Click <strong>Background remover</strong> in the left panel.</li>"
            "<li>Canva will automatically detect and remove the background.</li>"
            "<li>Use <strong>Restore</strong> and <strong>Erase</strong> brushes to refine edges.</li>"
            "<li>Click <strong>Apply</strong> when satisfied.</li>"
            "</ol><p>Note: Background remover is available on Canva Pro ($15/month) or with free trial.</p></section>",
            "<section class=\"seo\"><h2>Free alternative for Canva users</h2><ol>"
            "<li>Use our free <a href=\"/remove-background-from-image.html\">Remove Background</a> tool.</li>"
            "<li>Download the PNG with transparent background.</li>"
            "<li>Upload the PNG to Canva and use it in your designs.</li>"
            "</ol><p>This method is completely free and often produces better results than Canva's built-in tool.</p></section>",
            tools_list(),
            "<section class=\"seo\"><h2>Tips for Canva</h2><ul>"
            "<li>Use high-resolution images (at least 1000px wide) for best results.</li>"
            "<li>Images with clear subject-background contrast work best.</li>"
            "<li>Add a subtle shadow or border to make cutouts blend naturally.</li>"
            "<li>PNG format preserves transparency when downloading.</li>"
            "</ul></section>",
        ]
    elif "free" in k and "online" in k:
        sections = [
            "<section class=\"seo\"><p>Here are completely free, watermark-free ways to remove image backgrounds online without any software installation.</p></section>",
            "<section class=\"seo\"><h2>Free online background removal methods</h2><ol>"
            "<li><strong>Our Remove Background tool</strong> - No login, no watermark, unlimited use.</li>"
            "<li><strong>Google Photos</strong> - Magic Eraser feature (Android/Google One users).</li>"
            "<li><strong>Photopea</strong> - Free Photoshop alternative with background removal.</li>"
            "<li><strong>GIMP</strong> - Free desktop software with powerful selection tools.</li>"
            "</ol></section>",
            "<section class=\"seo\"><h2>Best free online method</h2><ol>"
            "<li>Visit our <a href=\"/remove-background-from-image.html\">Remove Background</a> tool.</li>"
            "<li>Upload your image (JPG, PNG, or WebP up to 10MB).</li>"
            "<li>Wait 5-10 seconds for AI processing.</li>"
            "<li>Download the PNG with transparent background.</li>"
            "</ol><p>This method is completely free, requires no registration, and produces professional results.</p></section>",
            tools_list(),
            "<section class=\"seo\"><h2>Why choose free online tools?</h2><ul>"
            "<li><strong>No software installation</strong> - Works in any web browser.</li>"
            "<li><strong>No watermarks</strong> - Download clean, professional results.</li>"
            "<li><strong>AI-powered</strong> - Automatic detection with manual refinement options.</li>"
            "<li><strong>Multiple formats</strong> - Support for JPG, PNG, and WebP images.</li>"
            "</ul></section>",
        ]
    elif "change" in k and "background" in k and "color" in k and "powerpoint" in k:
        sections = [
            "<section class=\"seo\"><p>PowerPoint offers several ways to change image background colors. Here are the most effective methods for different scenarios.</p></section>",
            "<section class=\"seo\"><h2>Method 1: Change background color of existing image</h2><ol>"
            "<li>Insert your image: Insert → Pictures → choose your file.</li>"
            "<li>Select the image and go to Picture Format → Color.</li>"
            "<li>Choose from preset color options or click Picture Color Options for custom colors.</li>"
            "<li>Adjust Brightness, Contrast, and Saturation as needed.</li>"
            "</ol><p>This method works best for images with solid or simple backgrounds.</p></section>",
            "<section class=\"seo\"><h2>Method 2: Remove background first, then add color</h2><ol>"
            "<li>Select your image → Picture Format → Remove Background.</li>"
            "<li>Use Mark Areas to Keep/Remove to refine the selection.</li>"
            "<li>Click Keep Changes to remove the background.</li>"
            "<li>Insert → Shapes → Rectangle, draw behind the image.</li>"
            "<li>Right-click the shape → Format Shape → Fill → choose your color.</li>"
            "</ol><p>This method gives you complete control over the background color.</p></section>",
            "<section class=\"seo\"><h2>Method 3: Use our online tool (best results)</h2><ol>"
            "<li>Visit our <a href=\"/change-image-background.html\">Change Background</a> tool.</li>"
            "<li>Upload your image and choose a background color.</li>"
            "<li>Download the result and insert it into PowerPoint.</li>"
            "</ol><p>This method produces the cleanest results and works with complex backgrounds.</p></section>",
            tools_list(),
            "<section class=\"seo\"><h2>Tips for PowerPoint backgrounds</h2><ul>"
            "<li>Use high-resolution images (at least 1920×1080) for presentations.</li>"
            "<li>Choose colors that contrast well with your text.</li>"
            "<li>Consider your presentation theme when selecting background colors.</li>"
            "<li>PNG format preserves transparency when importing into PowerPoint.</li>"
            "</ul></section>",
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
    """Legacy endpoint - now redirects to draft generation"""
    if CRON_TOKEN and token != CRON_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    # Redirect to the new draft generation system
    return await generate_blog_draft(token)


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

    # Canonicalize and deduplicate similar/alias posts (e.g., remove-background-from-image-* → remove-background-from-image)
    def canonicalize_slug(s: str) -> str:
        base = s.strip().lower()
        # Only collapse the exact near-duplicate variants, not tutorial-specific ones
        alias_map = {
            "remove-background-from-image": "remove-background-from-image",
            "remove-background-from-image-free": "remove-background-from-image",
            "remove-background-from-image-online": "remove-background-from-image",
            "remove-background-from-image-free-online": "remove-background-from-image",
            "remove-background-from-image-powerpoint": "remove-background-from-image",
            "remove-background-from-image-in-powerpoint": "remove-background-from-image",
        }
        return alias_map.get(base, base)

    dedup: dict[str, dict] = {}
    for p in items:
        slug = p.get("slug", "")
        canon = canonicalize_slug(slug)
        # Prefer an exact canonical entry if it exists; otherwise keep the most recent by date
        if canon not in dedup:
            dedup[canon] = {**p, "slug": canon, "title": p.get("title") or canon.replace('-', ' ').title()}
        else:
            # If existing is an alias but new one is truly canonical, replace
            if slug == canon and dedup[canon].get("slug") != canon:
                dedup[canon] = {**p, "slug": canon, "title": p.get("title") or canon.replace('-', ' ').title()}
            else:
                # Otherwise keep the one with the latest date
                try:
                    old_date = dedup[canon].get("date", "")
                    new_date = p.get("date", "")
                    if new_date and new_date > old_date:
                        dedup[canon] = {**p, "slug": canon, "title": p.get("title") or canon.replace('-', ' ').title()}
                except Exception:
                    pass
    items = list(dedup.values())
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
    # Redirect aliases to canonical to avoid duplicate content
    def canonicalize_slug(s: str) -> str:
        base = s.strip().lower()
        alias_map = {
            "remove-background-from-image-free": "remove-background-from-image",
            "remove-background-from-image-online": "remove-background-from-image",
            "remove-background-from-image-free-online": "remove-background-from-image",
        }
        return alias_map.get(base, base)

    canon = canonicalize_slug(slug)
    if canon != slug:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=f"/blog/{canon}.html", status_code=301)

    # Always render fresh from current template/sections
    # Use proper title formatting for specific slugs
    if slug == "remove-background-from-image-android":
        title = "How to Remove Background From Image on Android"
    else:
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


# ----------------------------
# Blog Management System
# ----------------------------

class BlogStatus(Enum):
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    PUBLISHED = "published"
    REJECTED = "rejected"

def init_blog_db():
    """Initialize SQLite database for blog management"""
    conn = sqlite3.connect('blog_management.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS blog_posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'draft',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            published_at TIMESTAMP NULL,
            author TEXT DEFAULT 'system',
            notes TEXT,
            metadata TEXT  -- JSON for additional data
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS blog_approvals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id INTEGER NOT NULL,
            action TEXT NOT NULL,  -- 'approve', 'reject', 'request_changes'
            approver TEXT NOT NULL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (post_id) REFERENCES blog_posts (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def get_db_connection():
    """Get database connection"""
    return sqlite3.connect('blog_management.db')

def check_local_only():
    """Check if we're in local development mode"""
    # Check if we're running on Cloud Run (production)
    if os.getenv("K_SERVICE") or os.getenv("ENVIRONMENT") == "production":
        raise HTTPException(status_code=404, detail="Admin interface not found")

# Initialize database on startup
init_blog_db()

@app.get("/api/blog/admin/drafts")
async def get_blog_drafts():
    """Get all draft, pending, approved, and rejected blog posts"""
    check_local_only()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, slug, title, status, created_at, updated_at, notes
        FROM blog_posts 
        WHERE status IN ('draft', 'pending_approval', 'approved', 'rejected')
        ORDER BY created_at DESC
    ''')
    
    drafts = []
    for row in cursor.fetchall():
        drafts.append({
            'id': row[0],
            'slug': row[1],
            'title': row[2],
            'status': row[3],
            'created_at': row[4],
            'updated_at': row[5],
            'notes': row[6]
        })
    
    conn.close()
    return {"drafts": drafts}

@app.get("/api/blog/admin/post/{post_id}")
async def get_blog_post(post_id: int):
    """Get specific blog post for editing"""
    check_local_only()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, slug, title, content, status, created_at, updated_at, 
               published_at, author, notes, metadata
        FROM blog_posts 
        WHERE id = ?
    ''', (post_id,))
    
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Post not found")
    
    post = {
        'id': row[0],
        'slug': row[1],
        'title': row[2],
        'content': row[3],
        'status': row[4],
        'created_at': row[5],
        'updated_at': row[6],
        'published_at': row[7],
        'author': row[8],
        'notes': row[9],
        'metadata': json.loads(row[10]) if row[10] else {}
    }
    
    conn.close()
    return post

@app.post("/api/blog/admin/post/{post_id}/update")
async def update_blog_post(post_id: int, request: Request):
    """Update blog post content"""
    check_local_only()
    data = await request.json()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Update post
    cursor.execute('''
        UPDATE blog_posts 
        SET title = ?, content = ?, updated_at = CURRENT_TIMESTAMP, notes = ?
        WHERE id = ?
    ''', (data.get('title'), data.get('content'), data.get('notes'), post_id))
    
    if cursor.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Post not found")
    
    conn.commit()
    conn.close()
    
    return {"success": True, "message": "Post updated successfully"}

@app.post("/api/blog/admin/post/{post_id}/approve")
async def approve_blog_post(post_id: int, request: Request):
    """Approve blog post for publishing"""
    check_local_only()
    data = await request.json()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Update post status
    cursor.execute('''
        UPDATE blog_posts 
        SET status = 'approved', updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    ''', (post_id,))
    
    if cursor.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Post not found")
    
    # Add approval record
    cursor.execute('''
        INSERT INTO blog_approvals (post_id, action, approver, notes)
        VALUES (?, 'approve', ?, ?)
    ''', (post_id, data.get('approver', 'admin'), data.get('notes', '')))
    
    conn.commit()
    conn.close()
    
    return {"success": True, "message": "Post approved successfully"}

@app.post("/api/blog/admin/post/{post_id}/reject")
async def reject_blog_post(post_id: int, request: Request):
    """Reject blog post"""
    check_local_only()
    data = await request.json()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Update post status
    cursor.execute('''
        UPDATE blog_posts 
        SET status = 'rejected', updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    ''', (post_id,))
    
    if cursor.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Post not found")
    
    # Add rejection record
    cursor.execute('''
        INSERT INTO blog_approvals (post_id, action, approver, notes)
        VALUES (?, 'reject', ?, ?)
    ''', (post_id, data.get('approver', 'admin'), data.get('notes', '')))
    
    conn.commit()
    conn.close()
    
    return {"success": True, "message": "Post rejected"}

@app.post("/api/blog/admin/post/{post_id}/publish")
async def publish_blog_post(post_id: int):
    """Publish approved blog post to live site"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get post data
    cursor.execute('''
        SELECT slug, title, content FROM blog_posts 
        WHERE id = ? AND status = 'approved'
    ''', (post_id,))
    
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Post not found or not approved")
    
    slug, title, content = row
    
    # Regenerate content with current title to ensure consistency
    fresh_content = render_article_html(title, slug, build_sections(title))
    
    # Publish to Google Cloud Storage (if configured)
    if BLOG_BUCKET:
        try:
            client = get_storage_client()
            save_article(client, slug, fresh_content)
            
            # Update post status
            cursor.execute('''
                UPDATE blog_posts 
                SET status = 'published', published_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (post_id,))
            
            conn.commit()
            conn.close()
            
            return {"success": True, "message": "Post published successfully"}
        except Exception as e:
            conn.close()
            raise HTTPException(status_code=500, detail=f"Publishing failed: {str(e)}")
    else:
        # For local development, try to publish to production automatically
        try:
            # Try to get storage client and publish to production
            if BLOG_BUCKET:
                client = get_storage_client()
                save_article(client, slug, fresh_content)
                message = "Post published to production successfully"
            else:
                message = "Post marked as published locally (BLOG_BUCKET not configured)"
            
            # Update post status regardless
            cursor.execute('''
                UPDATE blog_posts 
                SET content = ?, status = 'published', published_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (fresh_content, post_id))
            
            conn.commit()
            conn.close()
            
            return {"success": True, "message": message}
        except Exception as e:
            # If production publishing fails, still mark as published locally
            cursor.execute('''
                UPDATE blog_posts 
                SET content = ?, status = 'published', published_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (fresh_content, post_id))
            
            conn.commit()
            conn.close()
            
            return {"success": True, "message": f"Post marked as published locally (production publishing failed: {str(e)})"}

@app.post("/api/blog/admin/generate-draft")
async def generate_blog_draft(token: str = None):
    """Generate new blog post as draft (requires approval)"""
    check_local_only()
    # For local testing, allow without token
    if CRON_TOKEN and token != CRON_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    # Handle case when BLOG_BUCKET is not configured (local development)
    existing = set()
    if BLOG_BUCKET:
        try:
            client = get_storage_client()
            existing = list_existing_slugs(client)
        except Exception:
            existing = set()
    
    # Get existing draft slugs from database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT slug FROM blog_posts')
    db_slugs = {row[0] for row in cursor.fetchall()}
    conn.close()
    
    # Combine existing slugs
    all_existing = existing.union(db_slugs)
    
    seeds = [
        "remove background from image canva",
        "remove background from image for free online",
        "remove background from image in google slides",
        "remove background from image on iphone",
        "remove background from image in photoshop",
        "change image background color powerpoint",
    ]
    
    # Use exact keywords instead of generating variations
    picks = []
    for seed in seeds:
        slug = normalize_slug(seed)
        if slug not in all_existing:
            picks.append((seed, slug))
    
    created = []
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    for kw, slug in picks:
        title = kw.title()
        sections = build_sections(kw)
        html = render_article_html(title, slug, sections)
        
        # Save as draft in database
        cursor.execute('''
            INSERT INTO blog_posts (slug, title, content, status, author, notes)
            VALUES (?, ?, ?, 'draft', 'system', 'Auto-generated draft')
        ''', (slug, title, html))
        
        created.append({"slug": slug, "title": title, "status": "draft"})
    
    conn.commit()
    conn.close()
    
    return {"created": created, "message": "Drafts created - awaiting approval"}

@app.get("/api/blog/admin/published")
async def get_published_posts():
    """Get all published blog posts"""
    check_local_only()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, slug, title, status, created_at, published_at, author
        FROM blog_posts 
        WHERE status = 'published'
        ORDER BY published_at DESC
    ''')
    
    posts = []
    for row in cursor.fetchall():
        posts.append({
            'id': row[0],
            'slug': row[1],
            'title': row[2],
            'status': row[3],
            'created_at': row[4],
            'published_at': row[5],
            'author': row[6]
        })
    
    conn.close()
    return {"posts": posts}


@app.delete("/api/blog/admin/delete/{slug}")
async def delete_blog_post(slug: str):
    """Delete a blog post from Google Cloud Storage and update the blog index"""
    check_local_only()
    try:
        # Get storage client
        client = get_storage_client()
        bucket = client.bucket(BLOG_BUCKET)
        
        # Delete the HTML file
        blob_name = f"blog/{slug}.html"
        blob = bucket.blob(blob_name)
        
        if blob.exists():
            blob.delete()
            print(f"Deleted blog post: {slug}")
        else:
            print(f"Blog post not found: {slug}")
            return {"message": f"Blog post '{slug}' not found", "deleted": False}
        
        # Update the blog index by regenerating it
        await update_blog_index()
        
        return {"message": f"Blog post '{slug}' deleted successfully", "deleted": True}
        
    except Exception as e:
        print(f"Error deleting blog post {slug}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting blog post: {str(e)}")


async def update_blog_index():
    """Regenerate the blog index page after deleting posts"""
    try:
        client = get_storage_client()
        bucket = client.bucket(BLOG_BUCKET)
        
        # List all blog posts
        blobs = bucket.list_blobs(prefix="blog/")
        blog_posts = []
        
        for blob in blobs:
            if blob.name.endswith('.html') and blob.name != 'blog/index.html':
                # Extract slug from filename
                slug = blob.name.replace('blog/', '').replace('.html', '')
                
                # Get metadata
                blob.reload()
                metadata = blob.metadata or {}
                
                blog_posts.append({
                    'slug': slug,
                    'title': metadata.get('title', slug.replace('-', ' ').title()),
                    'date': metadata.get('date', '2025-09-20')
                })
        
        # Sort by date (newest first)
        blog_posts.sort(key=lambda x: x['date'], reverse=True)
        
        # Generate new index HTML
        index_html = f"""<!doctype html><html lang='en'><head>
<meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Image Editing Blog | Tutorials, Tips & How‑To Guides</title>
<meta name='description' content='Learn image editing: remove background, change colors, upscale, blur, enhance images. Free tutorials and guides.'>                                                  
<link rel='canonical' href='https://www.changeimageto.com/blog'>
<script type='application/ld+json'>{{"@context": "https://schema.org", "@type": "Blog", "name": "Image Editing Blog | Tutorials, Tips & How‑To Guides", "description": "Learn image editing: remove background, change colors, upscale, blur, enhance images. Free tutorials and guides.", "url": "https://www.changeimageto.com/blog"}}</script>                                                        
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
    <ul class='blog-list'>"""
        
        for post in blog_posts:
            index_html += f"<li><a href='/blog/{post['slug']}.html'>{post['title']}</a> <span class='date'>— {post['date']}</span></li>"
        
        index_html += """</ul>                                                                                                
  </main>
  <nav class='seo-links'><a href='/remove-background-from-image.html'>Remove Background</a><a href='/change-image-background.html'>Change Background</a><a href='/change-color-of-image.html'>Change Color</a><a href='/upscale-image.html'>AI Image Upscaler</a><a href='/blur-background.html'>Blur Background</a><a href='/enhance-image.html'>Enhance Image</a></nav>                                   
  <footer class='container footer'><p>Built for speed and quality. <a href='#' rel='nofollow'>Contact</a></p></footer>                                                                                
  <script src='/script.js?v=20250916-3' defer></script>
</body></html>"""
        
        # Upload updated index
        index_blob = bucket.blob("blog/index.html")
        index_blob.upload_from_string(index_html, content_type="text/html")
        
        print(f"Updated blog index with {len(blog_posts)} posts")
        
    except Exception as e:
        print(f"Error updating blog index: {e}")
        raise


@app.get("/blog-admin")
async def blog_admin():
    """Serve blog admin interface - Local development only"""
    # Only allow access in local development
    if os.getenv("ENVIRONMENT") == "production":
        raise HTTPException(status_code=404, detail="Admin interface not found")
    
    try:
        with open("frontend/blog-admin.html", "r", encoding="utf-8") as f:
            content = f.read()
        return Response(content=content, media_type="text/html")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Admin interface not found")


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


@app.post("/api/remove-text")
async def remove_text(
    request: Request,
    file: UploadFile = File(...),
):
    """Remove text from images using OCR detection and inpainting."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    
    log_user_action("text_removal_request", {
        "client_ip": client_ip,
        "user_agent": user_agent,
        "filename": file.filename,
    })
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Downscale for processing efficiency
        proc_image = downscale_image_if_needed(image)
        
        # Convert PIL to OpenCV format
        opencv_image = cv2.cvtColor(np.array(proc_image), cv2.COLOR_RGB2BGR)
        
        # Create mask for text regions
        text_mask = np.zeros(opencv_image.shape[:2], dtype=np.uint8)
        
        try:
            if not pytesseract:
                raise HTTPException(status_code=500, detail="Text detection not available")
                
            # Use Tesseract for text detection (more memory efficient)
            # Get text regions using Tesseract
            data = pytesseract.image_to_data(proc_image, output_type=pytesseract.Output.DICT)
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 50:  # Confidence threshold
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    if w > 10 and h > 10:  # Filter out very small detections
                        cv2.rectangle(text_mask, (x, y), (x + w, y + h), 255, -1)
                
        except Exception as e:
            log_user_action("text_detection_error", {"error": str(e)})
            raise HTTPException(status_code=500, detail=f"Text detection failed: {str(e)}")
        
        # Check if any text was detected
        if np.sum(text_mask) == 0:
            log_user_action("no_text_detected", {})
            # Return original image if no text detected
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            return Response(content=buf.getvalue(), media_type="image/png")
        
        # Gently expand and smooth the mask to avoid hard edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        text_mask = cv2.dilate(text_mask, kernel, iterations=2)
        text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Use OpenCV inpainting to remove text with a larger radius for better blending
        try:
            # Apply inpainting using Telea algorithm
            # Radius scaled to image size to avoid under/over inpainting
            max_side = max(opencv_image.shape[:2])
            dynamic_radius = int(max(5, min(15, max_side / 200)))
            inpainted = cv2.inpaint(opencv_image, text_mask, inpaintRadius=dynamic_radius, flags=cv2.INPAINT_TELEA)
            # Optional refinement pass with NS to improve structure continuity
            inpainted = cv2.inpaint(inpainted, text_mask, dynamic_radius, cv2.INPAINT_NS)

            # Feathered edge blend to reduce seams
            soft_mask = text_mask.copy().astype(np.float32) / 255.0
            soft_mask = cv2.GaussianBlur(soft_mask, (0, 0), sigmaX=3, sigmaY=3)
            inner = cv2.erode(text_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
            soft_mask[inner == 255] = 1.0
            soft_mask = np.clip(soft_mask, 0.0, 1.0)
            soft_mask_3 = np.repeat(soft_mask[:, :, None], 3, axis=2)
            blended = (soft_mask_3 * inpainted.astype(np.float32) + (1.0 - soft_mask_3) * opencv_image.astype(np.float32)).astype(np.uint8)
            inpainted = blended
            
            # Convert back to PIL Image
            result_image = Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
            
            # Resize back to original size if we downscaled
            if result_image.size != image.size:
                result_image = result_image.resize(image.size, Image.LANCZOS)
            
            log_user_action("text_removal_success", {
                "text_pixels_removed": int(np.sum(text_mask > 0))
            })
            
            buf = io.BytesIO()
            result_image.save(buf, format="PNG")
            return Response(content=buf.getvalue(), media_type="image/png")
            
        except Exception as e:
            log_user_action("inpainting_error", {"error": str(e)})
            raise HTTPException(status_code=500, detail=f"Text removal failed: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        log_user_action("text_removal_error", {"message": str(e)})
        raise HTTPException(status_code=500, detail=f"Text removal error: {str(e)}")


@app.post("/api/remove-painted-areas")
async def remove_painted_areas(
    request: Request,
    file: UploadFile = File(...),
    mask_data: str = Form(...),
):
    """Remove areas painted by user using inpainting."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    
    log_user_action("painted_areas_removal_request", {
        "client_ip": client_ip,
        "user_agent": user_agent,
        "filename": file.filename,
    })
    
    try:
        import base64
        
        # Load original image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Decode mask from base64
        try:
            # Remove data URL prefix if present
            if ',' in mask_data:
                mask_data = mask_data.split(',')[1]
            mask_bytes = base64.b64decode(mask_data)
            mask = Image.open(io.BytesIO(mask_bytes)).convert("L")  # Grayscale
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid mask data: {str(e)}")
        
        # Ensure mask and image are same size
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.LANCZOS)
        
        # Downscale for processing efficiency
        proc_image = downscale_image_if_needed(image)
        proc_mask = mask.resize(proc_image.size, Image.LANCZOS)
        
        # Convert to OpenCV format
        opencv_image = cv2.cvtColor(np.array(proc_image), cv2.COLOR_RGB2BGR)
        opencv_mask = np.array(proc_mask)
        
        # Create binary mask (white = remove, black = keep)
        # Lower threshold since our red overlay with 50% opacity becomes darker when converted to grayscale
        binary_mask = (opencv_mask > 50).astype(np.uint8) * 255
        
        # Check if any areas are marked for removal
        if np.sum(binary_mask) == 0:
            log_user_action("no_areas_marked_for_removal", {})
            # Return original image if no areas marked
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            return Response(content=buf.getvalue(), media_type="image/png")
        
        # Expand and smooth the mask to avoid harsh seams
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=2)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Radius scaled to image size and mask size
        max_side = max(opencv_image.shape[:2])
        mask_area = float(np.sum(binary_mask > 0))
        area_ratio = mask_area / float(binary_mask.shape[0] * binary_mask.shape[1] + 1e-6)
        base_radius = max(5, min(18, max_side / 180))
        # If very large region, increase radius slightly
        dynamic_radius = int(max(base_radius, 8 if area_ratio > 0.08 else base_radius))

        use_lama = os.getenv("USE_LAMA", "true").lower() in ("1","true","yes")
        large_hole = area_ratio > float(os.getenv("LAMA_MASK_THRESHOLD", "0.03"))
        
        # Debug logging
        log_user_action("mask_debug", {
            "mask_shape": opencv_mask.shape,
            "mask_min": int(opencv_mask.min()),
            "mask_max": int(opencv_mask.max()),
            "mask_mean": float(opencv_mask.mean()),
            "binary_mask_sum": int(np.sum(binary_mask)),
            "binary_mask_pixels": int(np.sum(binary_mask > 0)),
            "use_lama": use_lama,
            "large_hole": large_hole,
            "area_ratio": area_ratio,
            "lama_available": _get_lama_session() is not None
        })
        if use_lama and large_hole and _get_lama_session() is not None:
            result_base = lama_inpaint_onnx(opencv_image, binary_mask)
            log_user_action("inpaint_method", {"method": "lama_onnx"})
        else:
            # First pass: Telea
            result_telea = cv2.inpaint(opencv_image, binary_mask, dynamic_radius, cv2.INPAINT_TELEA)
            # Second pass: NS for structural continuity
            result_ns = cv2.inpaint(result_telea, binary_mask, dynamic_radius, cv2.INPAINT_NS)
            # Third pass: exemplar-based PatchMatch (if available) to improve textures on larger holes
            result_base = exemplar_inpaint_patchmatch(result_ns, binary_mask)
            log_user_action("inpaint_method", {"method": "opencv_fallback", "radius": dynamic_radius})

        # Feather edge to reduce halos
        soft = binary_mask.astype(np.float32) / 255.0
        soft = cv2.GaussianBlur(soft, (0, 0), sigmaX=4, sigmaY=4)
        inner = cv2.erode(binary_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        soft[inner == 255] = 1.0
        soft = np.clip(soft, 0.0, 1.0)
        soft3 = np.repeat(soft[:, :, None], 3, axis=2)
        # Poisson-like seamless cloning along the boundary when available
        try:
            # Build a tight fill region around mask boundary
            boundary = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)), iterations=1)
            center = _mask_center(binary_mask)
            mixed = cv2.seamlessClone(result_base, opencv_image, boundary, center, cv2.MIXED_CLONE)
            base_for_blend = mixed
        except Exception:
            base_for_blend = result_base

        result = (soft3 * base_for_blend.astype(np.float32) + (1.0 - soft3) * opencv_image.astype(np.float32)).astype(np.uint8)
        
        # Convert back to PIL
        result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        
        # Resize back to original size if we downscaled
        if result_image.size != image.size:
            result_image = result_image.resize(image.size, Image.LANCZOS)
        
        log_user_action("painted_areas_removal_success", {
            "painted_pixels_removed": int(np.sum(binary_mask > 0))
        })
        
        buf = io.BytesIO()
        result_image.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
        
    except HTTPException:
        raise
    except Exception as e:
        log_user_action("painted_areas_removal_error", {"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Painted areas removal failed: {str(e)}")


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
    
    async def process_single_resize(i, file):
        """Process a single image resize and return result or error."""
        try:
            # Validate file type
            if not file.content_type or not file.content_type.startswith("image/"):
                return f"File {i+1} ({file.filename}): Not an image file"
            
            # Read and process image
            contents = await file.read()
            if len(contents) > 10 * 1024 * 1024:  # 10MB limit per file
                return f"File {i+1} ({file.filename}): File too large (max 10MB)"
            
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
            
            return {
                'filename': filename,
                'data': output_data,
                'original_size': f"{image.width}x{image.height}",
                'new_size': f"{new_width}x{new_height}"
            }
            
        except Exception as e:
            return f"File {i+1} ({file.filename}): {str(e)}"
    
    try:
        # Process images in parallel with bounded concurrency
        tasks = [process_single_resize(i, file) for i, file in enumerate(files)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Separate successful results from errors
        for result in results:
            if isinstance(result, str):  # Error message
                errors.append(result)
            elif isinstance(result, dict):  # Successful resize
                processed_images.append(result)
            else:  # Exception
                errors.append(f"Unexpected error: {str(result)}")
        
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


@app.post("/api/bulk-convert-format")
async def bulk_convert_format(
    request: Request,
    files: List[UploadFile] = File(...),
    target_format: str = Form("png"),
    transparent: bool = Form(False),
    quality: int = Form(90),
):
    """Convert multiple images to a different format and return as a ZIP file."""
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    
    # Validate inputs
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 images allowed per batch")
    
    if quality < 1 or quality > 100:
        raise HTTPException(status_code=400, detail="Quality must be between 1 and 100")
    
    supported_targets = {"png", "jpg", "webp", "bmp", "gif", "tiff", "ico", "ppm", "pgm"}
    if target_format not in supported_targets:
        raise HTTPException(status_code=400, detail=f"Unsupported target_format. Use one of: {sorted(supported_targets)}")
    
    log_user_action("bulk_convert_started", {
        "client_ip": client_ip,
        "user_agent": user_agent,
        "file_count": len(files),
        "target_format": target_format,
        "transparent": transparent,
        "quality": quality
    })
    
    processed_images = []
    errors = []
    
    async def process_single_convert(i, file):
        """Process a single image format conversion and return result or error."""
        try:
            # Validate file type
            if not file.content_type or not file.content_type.startswith("image/"):
                return f"File {i+1} ({file.filename}): Not an image file"
            
            # Read and process image
            contents = await file.read()
            if len(contents) > 10 * 1024 * 1024:  # 10MB limit per file
                return f"File {i+1} ({file.filename}): File too large (max 10MB)"
            
            image = Image.open(io.BytesIO(contents))
            
            # Decide mode based on transparency setting and target
            supports_alpha = target_format in {"png", "webp", "tiff", "ico", "gif"}
            keep_alpha = bool(transparent and supports_alpha)
            
            # Convert image based on format requirements
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
                # Flatten onto opaque white background for formats w/o alpha
                if image.mode in ("RGBA", "LA"):
                    background = Image.new("RGB", image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1])
                    converted = background
                else:
                    converted = image.convert("RGB")
            
            # Prepare save parameters
            params = {}
            if target_format == "jpg":
                params.update({"quality": max(1, min(int(quality), 100)), "optimize": True})
                save_format = "JPEG"
            elif target_format == "webp":
                params.update({"quality": max(1, min(int(quality), 100))})
                save_format = "WEBP"
            elif target_format == "bmp":
                save_format = "BMP"
            elif target_format == "gif":
                if keep_alpha:
                    converted = converted.convert("RGBA")
                    palette_image = converted.convert("P", palette=Image.ADAPTIVE, colors=255)
                    alpha = converted.split()[-1]
                    mask = Image.eval(alpha, lambda a: 255 if a <= 1 else 0)
                    palette_image.info['transparency'] = 255
                    palette_image.paste(255, None, mask)
                    converted = palette_image
                else:
                    converted = converted.convert("P", palette=Image.ADAPTIVE, colors=256)
                params.update({"optimize": True, "save_all": False})
                save_format = "GIF"
            elif target_format == "tiff":
                params.update({"compression": "tiff_lzw"})
                save_format = "TIFF"
            elif target_format == "ico":
                # ICO must be 256x256
                size = 256
                img_rgba = converted.convert("RGBA")
                min_side = min(img_rgba.width, img_rgba.height)
                scale = size / float(min_side)
                resized = img_rgba.resize((max(1, int(round(img_rgba.width * scale))),
                                           max(1, int(round(img_rgba.height * scale)))), Image.LANCZOS)
                left = (resized.width - size) // 2
                top = (resized.height - size) // 2
                converted = resized.crop((left, top, left + size, top + size))
                save_format = "ICO"
                params.update({"sizes": [(256, 256)]})
            elif target_format in {"ppm", "pgm"}:
                if target_format == "ppm":
                    converted = converted.convert("RGB")
                else:
                    converted = converted.convert("L")
                save_format = "PPM"
                params.update({"bits": 8})
            else:
                save_format = "PNG"
            
            # Save to bytes
            output_buffer = io.BytesIO()
            converted.save(output_buffer, format=save_format, **params)
            output_data = output_buffer.getvalue()
            
            # Store filename and data
            filename = file.filename or f"image_{i+1}.{target_format}"
            if not filename.lower().endswith(f'.{target_format}'):
                name_without_ext = filename.rsplit('.', 1)[0] if '.' in filename else filename
                filename = f"{name_without_ext}.{target_format}"
            
            return {
                'filename': filename,
                'data': output_data,
                'original_size': f"{image.width}x{image.height}",
                'new_format': target_format
            }
            
        except Exception as e:
            return f"File {i+1} ({file.filename}): {str(e)}"
    
    try:
        # Process images in parallel with bounded concurrency
        tasks = [process_single_convert(i, file) for i, file in enumerate(files)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Separate successful results from errors
        for result in results:
            if isinstance(result, str):  # Error message
                errors.append(result)
            elif isinstance(result, dict):  # Successful conversion
                processed_images.append(result)
            else:  # Exception
                errors.append(f"Unexpected error: {str(result)}")
        
        if not processed_images:
            raise HTTPException(status_code=400, detail="No images could be processed. " + "; ".join(errors))
        
        # Create ZIP file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for img_data in processed_images:
                zip_file.writestr(img_data['filename'], img_data['data'])
        
        zip_data = zip_buffer.getvalue()
        
        log_user_action("bulk_convert_completed", {
            "processed_count": len(processed_images),
            "error_count": len(errors),
            "target_format": target_format,
            "zip_size_bytes": len(zip_data),
            "zip_size_mb": round(len(zip_data) / (1024 * 1024), 2)
        })
        
        # Return ZIP file
        headers = {
            "Content-Disposition": f"attachment; filename=converted_images_{target_format}.zip",
            "Content-Type": "application/zip"
        }
        
        return Response(content=zip_data, headers=headers, media_type="application/zip")
        
    except HTTPException:
        raise
    except Exception as e:
        log_user_action("bulk_convert_error", {"message": str(e)})
        raise HTTPException(status_code=500, detail=f"Bulk convert error: {str(e)}")


def detect_image_quality(image_array):
    """Analyze image quality and return detailed metrics."""
    try:
        # Convert to grayscale for analysis
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        quality_metrics = {}
        issues = []
        overall_score = 100
        
        # Common stats
        height, width = gray.shape
        total_pixels = float(height * width)
        gray_float = gray.astype(np.float64)
        gray_std = float(np.std(gray_float))
        gray_mean = float(np.mean(gray_float))

        # 1) Blur detection (normalized Laplacian + Tenengrad + depth-of-field analysis)
        lap = cv2.Laplacian(gray_float, cv2.CV_64F)
        lap_var = float(lap.var())
        sobel_x = cv2.Sobel(gray_float, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_float, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = np.hypot(sobel_x, sobel_y)
        ten_var = float(tenengrad.var())
        texture_var = float(np.var(gray_float)) + 1e-6
        lap_norm = lap_var / texture_var
        ten_norm = ten_var / texture_var
        blur_index = 0.5 * lap_norm + 0.5 * ten_norm  # higher is sharper
        
        # Depth-of-field detection: check if center is sharper than edges (intentional blur)
        center_h, center_w = height // 2, width // 2
        center_size = min(height, width) // 3  # Center region size
        y1, y2 = center_h - center_size//2, center_h + center_size//2
        x1, x2 = center_w - center_size//2, center_w + center_size//2
        y1, y2 = max(0, y1), min(height, y2)
        x1, x2 = max(0, x1), min(width, x2)
        
        # Calculate sharpness in center vs edges
        center_lap = lap[y1:y2, x1:x2]
        center_ten = tenengrad[y1:y2, x1:x2]
        center_sharpness = float(np.mean(center_lap)) + float(np.mean(center_ten))
        
        # Edge regions (corners and borders)
        edge_regions = []
        if y1 > 0: edge_regions.append(lap[:y1, :])  # top
        if y2 < height: edge_regions.append(lap[y2:, :])  # bottom
        if x1 > 0: edge_regions.append(lap[:, :x1])  # left
        if x2 < width: edge_regions.append(lap[:, x2:])  # right
        
        if edge_regions:
            edge_sharpness = float(np.mean([np.mean(region) for region in edge_regions]))
            # If center is significantly sharper than edges, likely intentional DOF blur
            dof_ratio = center_sharpness / (edge_sharpness + 1e-6)
            if dof_ratio > 2.0:  # Center is 2x sharper than edges
                # Boost blur score for intentional depth-of-field
                blur_index *= 1.3
                blur_score = max(0.0, min(100.0, (blur_index / 1.2) * 100.0))
            else:
                blur_score = max(0.0, min(100.0, (blur_index / 1.2) * 100.0))
        else:
            blur_score = max(0.0, min(100.0, (blur_index / 1.2) * 100.0))
        
        if blur_score < 40:  # More lenient threshold for decent images
            issues.append("blur")
        quality_metrics["blur_score"] = float(round(blur_score, 1))

        # 2) Noise detection (resolution-aware + texture-aware filtering)
        hp_kernel = np.array([[-1, -1, -1],
                              [-1,  8, -1],
                              [-1, -1, -1]], dtype=np.float64)
        residual = cv2.filter2D(gray_float, -1, hp_kernel, borderType=cv2.BORDER_REFLECT)
        
        # Resolution-aware noise threshold (higher res = more detail = higher noise threshold)
        resolution_factor = min(2.0, max(0.5, total_pixels / 500000.0))  # Scale based on resolution
        
        # Build flat-region mask: low edges and low local variance
        edges_mask = cv2.Canny(gray.astype(np.uint8), 50, 150) == 0
        mean = cv2.blur(gray_float, (5, 5))
        sqmean = cv2.blur(gray_float * gray_float, (5, 5))
        local_var = np.maximum(0.0, sqmean - mean * mean)
        
        # Adjust texture threshold based on resolution
        texture_threshold = 50.0 * resolution_factor
        flats_mask = (local_var < texture_threshold) & edges_mask
        
        if np.any(flats_mask):
            residual_std = float(np.std(residual[flats_mask]))
        else:
            residual_std = float(np.std(residual))
        
        snr = (gray_mean + 1e-6) / (gray_std + 1e-6)
        
        # Contrast-aware SNR scoring (high contrast = good, not noise)
        # Calculate contrast ratio to distinguish good contrast from noise
        contrast_ratio = gray_std / (gray_mean + 1e-6)
        
        # For high-contrast images (product photos), adjust SNR scoring
        if contrast_ratio > 0.3:  # High contrast image
            # Boost SNR score for high-contrast images (good contrast, not noise)
            snr_score = max(0.0, min(100.0, (snr / 8.0) * 100.0))  # More lenient threshold
        else:
            # Normal SNR scoring for low-contrast images
            snr_score = max(0.0, min(100.0, (snr / 10.0) * 100.0))
        # Adjust residual penalty based on resolution
        residual_threshold = 50.0 * resolution_factor
        residual_penalty = max(0.0, min(100.0, (residual_std / residual_threshold) * 100.0))
        
        # Combine scores with resolution awareness
        noise_score = max(0.0, min(100.0, 0.7 * snr_score + 0.3 * (100.0 - residual_penalty)))
        
        # Adjust threshold based on resolution (high-res images get more lenient threshold)
        noise_threshold = max(25.0, 35.0 - (resolution_factor - 1.0) * 10.0)
        
        if noise_score < noise_threshold:
            issues.append("noise")
        quality_metrics["noise_score"] = float(round(noise_score, 1))

        # 3) Pixelation detection (FFT grid peaks + edge density + SSIM smoothness check)
        # Downsample to speed up FFT while keeping grid artifacts
        max_side = 512
        scale = max(1.0, max(height, width) / max_side)
        if scale > 1.0:
            small = cv2.resize(gray_float, (int(round(width / scale)), int(round(height / scale))), interpolation=cv2.INTER_AREA)
        else:
            small = gray_float
        # FFT magnitude
        f = np.fft.fftshift(np.fft.fft2(small))
        mag = np.abs(f)
        # Ignore DC and very low frequencies
        h, w = small.shape
        cy, cx = h // 2, w // 2
        low_radius = max(3, int(0.02 * min(h, w)))
        yy, xx = np.ogrid[:h, :w]
        mask_low = (yy - cy) ** 2 + (xx - cx) ** 2 <= low_radius ** 2
        mag_filt = mag.copy()
        mag_filt[mask_low] = 0
        # Estimate grid energy along 8-pixel periodicity harmonics
        # Translate pixel-domain 8px blocks to frequency-domain ~w/8 and h/8 bands
        bands = []
        for k in (1, 2, 3):
            vx = int(round((w / 8.0) * k))
            vy = int(round((h / 8.0) * k))
            bands.append((vx, 0))
            bands.append((0, vy))
        band_radius = max(1, int(0.01 * min(h, w)))
        grid_energy = 0.0
        for bx, by in bands:
            if bx != 0:
                grid_energy += float(mag_filt[:, max(0, cx - bx - band_radius):min(w, cx - bx + band_radius)].sum())
                grid_energy += float(mag_filt[:, max(0, cx + bx - band_radius):min(w, cx + bx + band_radius)].sum())
            if by != 0:
                grid_energy += float(mag_filt[max(0, cy - by - band_radius):min(h, cy - by + band_radius), :].sum())
                grid_energy += float(mag_filt[max(0, cy + by - band_radius):min(h, cy + by + band_radius), :].sum())
        total_hf_energy = float(mag_filt.sum()) + 1e-6
        grid_ratio = grid_energy / total_hf_energy
        # Edge density
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        edge_density = float(np.sum(edges > 0) / total_pixels)
        # Pixelation score: penalize when grid_ratio high and edges sparse
        pixelation_penalty = max(0.0, min(100.0, (grid_ratio / 0.3) * 100.0))  # 0.3 ~ strong grid
        # SSIM smoothness check via 2x bicubic upsample then downsample
        try:
            up2 = cv2.resize(gray.astype(np.uint8), (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
            down2 = cv2.resize(up2, (width, height), interpolation=cv2.INTER_AREA)
            ssim_val = float(ssim(gray.astype(np.uint8), down2, data_range=255))
        except Exception:
            ssim_val = 1.0
        ssim_change = 1.0 - ssim_val
        if edge_density < 0.06 and grid_ratio > 0.12 and ssim_change > 0.15:
            issues.append("pixelation")
        else:
            # dampen penalty if SSIM does not indicate strong blockiness reduction
            pixelation_penalty *= 0.5
        pixelation_score = max(0.0, min(100.0, 100.0 - pixelation_penalty))
        quality_metrics["pixelation_score"] = float(round(pixelation_score, 1))

        # 4) Exposure & lighting (histogram balance + white background detection)
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist.astype(np.float64)
        hist /= (hist.sum() + 1e-6)
        dark_pixels = float(np.sum(hist[:85]))
        mid_pixels = float(np.sum(hist[85:170]))
        bright_pixels = float(np.sum(hist[170:]))
        
        # White background detection (common in product photos)
        very_bright_pixels = float(np.sum(hist[200:]))  # Very bright pixels (200-255)
        white_background_detected = very_bright_pixels > 0.4  # 40%+ very bright pixels
        
        # balance score
        exposure_score = 100.0
        if dark_pixels > 0.60:  # More lenient threshold for decent images
            issues.append("underexposed")
            exposure_score = max(0.0, 100.0 - (dark_pixels - 0.50) * 200.0)
        elif bright_pixels > 0.60:  # More lenient threshold for decent images
            # Special handling for white background product photos
            if white_background_detected:
                # For white backgrounds, only flag if extremely overexposed
                if bright_pixels > 0.80:  # 80%+ bright pixels
                    issues.append("overexposed")
                    exposure_score = max(0.0, 100.0 - (bright_pixels - 0.70) * 150.0)
                else:
                    # White background is acceptable, don't penalize heavily
                    exposure_score = max(80.0, 100.0 - (bright_pixels - 0.60) * 50.0)
            else:
                issues.append("overexposed")
                exposure_score = max(0.0, 100.0 - (bright_pixels - 0.50) * 200.0)
        elif mid_pixels < 0.25:
            exposure_score = max(0.0, mid_pixels * 400.0)
        # entropy (0..1)
        nz = hist[hist > 0]
        entropy = float(-(nz * np.log2(nz)).sum() / np.log2(256))
        entropy_score = max(0.0, min(100.0, entropy * 100.0))
        # RMS contrast
        contrast_score = max(0.0, min(100.0, (gray_std / 255.0) * 100.0))
        # Clipping measure at 0/255
        zeros_frac = float(np.mean(gray == 0))
        fulls_frac = float(np.mean(gray == 255))
        clip_frac = zeros_frac + fulls_frac
        if clip_frac > 0.15:  # More lenient threshold for decent images
            issues.append("clipped")
            exposure_score = min(exposure_score, max(0.0, 100.0 - (clip_frac - 0.10) * 300.0))
        exposure_combined = 0.5 * exposure_score + 0.25 * entropy_score + 0.25 * contrast_score
        quality_metrics["exposure_score"] = float(round(exposure_combined, 1))

        # 5) Resolution & usability (with aspect ratio factor)
        if total_pixels < 100000:  # ~316x316
            issues.append("low_resolution")
            resolution_score = max(0.0, (total_pixels / 100000.0) * 100.0)
        elif total_pixels < 400000:  # ~632x632
            resolution_score = 80.0
        else:
            resolution_score = 100.0
        aspect_ratio = max(width / float(height), height / float(width))
        if aspect_ratio > 3.0:
            # penalize extreme panoramas
            penalty = min(30.0, (aspect_ratio - 3.0) * 10.0)
            resolution_score = max(0.0, resolution_score - penalty)
        # Couple resolution with blur (high res but soft images shouldn't get full credit)
        if resolution_score > 80.0 and quality_metrics["blur_score"] < 50.0:
            res_cap = quality_metrics["blur_score"] + 20.0
            resolution_score = min(resolution_score, res_cap)
        quality_metrics["resolution_score"] = float(round(resolution_score, 1))

        # 6) Compression artifacts (blockiness across 8x8 borders)
        # Compute border differences at multiples of 8
        block = 8
        # vertical borders energy
        vb = 0.0
        for x in range(block, width, block):
            vb += float(np.sum(np.abs(gray_float[:, x-1] - gray_float[:, x % width])))
        # horizontal borders energy
        hb = 0.0
        for y in range(block, height, block):
            hb += float(np.sum(np.abs(gray_float[y-1, :] - gray_float[y % height, :])))
        border_energy = vb + hb
        # baseline intra-block energy (differences away from borders)
        # use shifted differences to approximate
        intra_h = float(np.sum(np.abs(gray_float[:, 1:] - gray_float[:, :-1]))) + 1e-6
        intra_v = float(np.sum(np.abs(gray_float[1:, :] - gray_float[:-1, :]))) + 1e-6
        intra_total = intra_h + intra_v
        blockiness_ratio = float(border_energy) / float(intra_total)
        compression_score = max(0.0, min(100.0, 100.0 - (blockiness_ratio / 0.12) * 100.0))  # 0.12 is noticeable blockiness
        # Texture-aware damping for flat logos/solids
        if entropy_score < 20.0:
            compression_score = 100.0
            issues = [iss for iss in issues if iss != "compression_artifacts"]
        elif compression_score < 50.0:  # More lenient threshold for decent images
            issues.append("compression_artifacts")
        quality_metrics["compression_score"] = float(round(compression_score, 1))

        # Tiny image adjustments: soften harsh penalties for very small images
        min_side = min(height, width)
        if min_side < 128:
            # Relax blur/exposure for icons/thumbnails where metrics are unreliable
            quality_metrics["blur_score"] = float(max(quality_metrics["blur_score"], 40.0))
            quality_metrics["exposure_score"] = float(max(quality_metrics["exposure_score"], 40.0))
            # Do not flag exposure issues for tiny images
            issues = [iss for iss in issues if iss not in ("underexposed", "overexposed")]

        # Entropy-based blur relaxation for flat backgrounds
        if entropy_score < 30.0:
            pre = quality_metrics["blur_score"]
            quality_metrics["blur_score"] = float(max(quality_metrics["blur_score"], 60.0))
            if pre < 60.0 and "blur" in issues and quality_metrics["blur_score"] >= 60.0:
                issues = [iss for iss in issues if iss != "blur"]

        # Overall scoring (rebalanced for new metrics)
        weights = {
            'blur_score': 0.25,
            'resolution_score': 0.25,
            'exposure_score': 0.15,  # Reduced weight since it's often subjective
            'noise_score': 0.15,
            'compression_score': 0.15,  # Increased weight
            'pixelation_score': 0.05,
        }
        overall_score = (
            quality_metrics['blur_score'] * weights['blur_score'] +
            quality_metrics['resolution_score'] * weights['resolution_score'] +
            quality_metrics['exposure_score'] * weights['exposure_score'] +
            quality_metrics['noise_score'] * weights['noise_score'] +
            quality_metrics['compression_score'] * weights['compression_score'] +
            quality_metrics['pixelation_score'] * weights['pixelation_score']
        )
        quality_metrics["overall_score"] = round(float(overall_score), 1)
        quality_metrics["issues"] = issues
        quality_metrics["dimensions"] = f"{width}x{height}"
        
        return quality_metrics
        
    except Exception as e:
        return {
            "overall_score": 0,
            "issues": ["analysis_failed"],
            "error": str(e),
            "blur_score": 0,
            "noise_score": 0,
            "pixelation_score": 0,
            "exposure_score": 0,
            "resolution_score": 0,
            "compression_score": 0
        }


@app.post("/api/bulk-quality-check")
async def bulk_quality_check(
    request: Request,
    files: List[UploadFile] = File(...),
):
    """Analyze quality of multiple images and return detailed reports."""
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    
    # Validate inputs
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 images allowed per batch")
    
    log_user_action("bulk_quality_check_started", {
        "client_ip": client_ip,
        "user_agent": user_agent,
        "file_count": len(files)
    })
    
    analyzed_images = []
    errors = []
    
    async def process_single_image(i, file):
        """Process a single image file and return result or error."""
        try:
            # Validate file type
            if not file.content_type or not file.content_type.startswith("image/"):
                return f"File {i+1} ({file.filename}): Not an image file"
            
            # Read and process image
            contents = await file.read()
            if len(contents) > 10 * 1024 * 1024:  # 10MB limit per file
                return f"File {i+1} ({file.filename}): File too large (max 10MB)"
            
            # Convert to OpenCV format
            nparr = np.frombuffer(contents, np.uint8)
            image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image_array is None:
                return f"File {i+1} ({file.filename}): Could not decode image"
            
            # Convert BGR to RGB for analysis
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            
            # Analyze image quality (CPU-bound, run in thread pool)
            async with PROCESS_SEM:
                quality_metrics = await asyncio.to_thread(detect_image_quality, image_array)
                
                # Determine quality category
                overall_score = quality_metrics["overall_score"]
                if overall_score >= 80:
                    category = "good"
                elif overall_score >= 60:
                    category = "fair"
                else:
                    category = "poor"
                
                # Generate repair suggestions
                repair_links = []
                for issue in quality_metrics.get("issues", []):
                    if issue == "blur":
                        repair_links.append({"tool": "enhance", "url": "/enhance-image.html", "description": "Enhance Image"})
                    elif issue == "noise":
                        repair_links.append({"tool": "enhance", "url": "/enhance-image.html", "description": "Enhance Image"})
                    elif issue == "pixelation":
                        repair_links.append({"tool": "upscale", "url": "/upscale-image.html", "description": "AI Image Upscaler"})
                    elif issue == "overexposed":
                        repair_links.append({"tool": "enhance", "url": "/enhance-image.html", "description": "Enhance Image"})
                    elif issue == "underexposed":
                        repair_links.append({"tool": "enhance", "url": "/enhance-image.html", "description": "Enhance Image"})
                    elif issue == "low_resolution":
                        repair_links.append({"tool": "upscale", "url": "/upscale-image.html", "description": "AI Image Upscaler"})
                    elif issue == "compression_artifacts":
                        repair_links.append({"tool": "enhance", "url": "/enhance-image.html", "description": "Enhance Image"})
                
                return {
                    'filename': file.filename or f"image_{i+1}",
                    'quality_metrics': quality_metrics,
                    'category': category,
                    'repair_links': repair_links,
                    'file_size_mb': round(len(contents) / (1024 * 1024), 2)
                }
                
        except Exception as e:
            return f"File {i+1} ({file.filename}): {str(e)}"
    
    try:
        # Process images in parallel with bounded concurrency
        tasks = [process_single_image(i, file) for i, file in enumerate(files)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Separate successful results from errors
        for result in results:
            if isinstance(result, str):  # Error message
                errors.append(result)
            elif isinstance(result, dict):  # Successful analysis
                analyzed_images.append(result)
            else:  # Exception
                errors.append(f"Unexpected error: {str(result)}")
        
        if not analyzed_images:
            raise HTTPException(status_code=400, detail="No images could be analyzed. " + "; ".join(errors))
        
        # Categorize images
        good_images = [img for img in analyzed_images if img['category'] == 'good']
        fair_images = [img for img in analyzed_images if img['category'] == 'fair']
        poor_images = [img for img in analyzed_images if img['category'] == 'poor']
        
        log_user_action("bulk_quality_check_completed", {
            "total_analyzed": len(analyzed_images),
            "good_count": len(good_images),
            "fair_count": len(fair_images),
            "poor_count": len(poor_images),
            "error_count": len(errors)
        })
        
        return {
            "total_images": len(analyzed_images),
            "good_images": len(good_images),
            "fair_images": len(fair_images),
            "poor_images": len(poor_images),
            "error_count": len(errors),
            "images": analyzed_images,
            "errors": errors
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_user_action("bulk_quality_check_error", {"message": str(e)})
        raise HTTPException(status_code=500, detail=f"Quality check error: {str(e)}")


