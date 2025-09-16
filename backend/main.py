from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageEnhance, ImageOps
import io
import logging
import asyncio
import json
import os
from datetime import datetime
from typing import Optional
import numpy as np

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
    "product": "u2net",
    "portrait": "u2net_human_seg",
}

_sessions_cache = {}
def downscale_image_if_needed(image: Image.Image, max_side: int = int(os.getenv("MAX_IMAGE_SIDE", "3000"))) -> Image.Image:
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

def get_session_for_category(category: str):
    category = category.lower()
    model_name = MODEL_NAME_FOR_CATEGORY.get(category)
    if not model_name:
        raise HTTPException(status_code=400, detail=f"Unsupported category: {category}")
    if model_name not in _sessions_cache:
        _sessions_cache[model_name] = new_session(model_name)
    return _sessions_cache[model_name]


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

        # Downscale to protect memory
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
            
        # Optional solid background color compositing
        if bg_color:
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
            background = Image.new('RGBA', result.size, (r, g, b, 255))
            background.alpha_composite(result)
            result = background.convert('RGBA')
            
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
