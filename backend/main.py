from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from rembg import remove, new_session

app = FastAPI(title="BG Remover", description="Simple background removal API", version="1.0.0")

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
    file: UploadFile = File(...),
    category: str = Form("product"),
    bg_color: str | None = Form(None),
):
    try:
        session = get_session_for_category(category)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model init error: {str(e)}")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGBA")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
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
                raise HTTPException(status_code=400, detail="Invalid bg_color; use hex like #ffffff")
            r = int(col[0:2], 16); g = int(col[2:4], 16); b = int(col[4:6], 16)
            background = Image.new('RGBA', result.size, (r, g, b, 255))
            background.alpha_composite(result)
            result = background.convert('RGBA')
        output_io = io.BytesIO()
        result.save(output_io, format="PNG")
        output_bytes = output_io.getvalue()
        return Response(content=output_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/")
async def healthcheck():
    return {"status": "ok"}
