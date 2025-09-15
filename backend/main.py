from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import logging
import json
import os
from datetime import datetime
from typing import Optional

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
        
        result = remove(image, session=session, alpha_matting=True, alpha_matting_foreground_threshold=240, alpha_matting_background_threshold=10, alpha_matting_erode_size=10, post_process_mask=True)
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
