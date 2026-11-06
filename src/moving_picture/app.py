import base64
import uuid
from collections import OrderedDict

import anthropic
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import ROOT, load_env
from .analyze import ask_claude
from .rig import build_rig
from .vision import fill_behind, load_photo, straighten

load_env()
WEB = ROOT / "web"
KEEP_SCANS = 20

app = FastAPI()


@app.middleware("http")
async def no_stale_pages(request, call_next):
    # Without this the browser may keep running an old main.js after the app is updated.
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache"
    return response


scans: OrderedDict[str, np.ndarray] = OrderedDict()  # single-user local app: keep photos in memory


def data_url(img: np.ndarray, fmt: str) -> str:
    code = cv2.COLOR_RGBA2BGRA if img.shape[2] == 4 else cv2.COLOR_RGB2BGR
    ok, buf = cv2.imencode(fmt, cv2.cvtColor(img, code), [cv2.IMWRITE_JPEG_QUALITY, 92])
    mime = "image/png" if fmt == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buf.tobytes()).decode()


def get_scan(scan_id: str) -> np.ndarray:
    if scan_id not in scans:
        raise HTTPException(404, "This picture has expired, please add it again")
    return scans[scan_id]


@app.post("/api/scan")
async def scan(file: UploadFile = File(...)):
    try:
        image = straighten(load_photo(await file.read()))
    except Exception:
        raise HTTPException(422, "Couldn't open this image, try another one")
    scan_id = uuid.uuid4().hex
    scans[scan_id] = image
    while len(scans) > KEEP_SCANS:
        scans.popitem(last=False)
    h, w = image.shape[:2]
    return {"id": scan_id, "width": w, "height": h, "image": data_url(image, ".jpg")}


class ScanRef(BaseModel):
    id: str


@app.post("/api/analyze")
def analyze(ref: ScanRef):
    image = get_scan(ref.id)
    try:
        return ask_claude(image)
    except anthropic.AuthenticationError:
        raise HTTPException(500, "Claude API key is missing or invalid (CLAUDE_API_KEY in .env)")
    except anthropic.APIConnectionError:
        raise HTTPException(502, "Can't reach Claude, check the internet connection")
    except anthropic.APIStatusError as e:
        raise HTTPException(502, f"Claude error: {e.message}")
    except RuntimeError as e:
        raise HTTPException(422, str(e))


class Joint(BaseModel):
    name: str
    x: int
    y: int
    parent: str


class Character(BaseModel):
    id: str
    name: str
    bbox: list[int]
    joints: list[Joint]


class RigRequest(BaseModel):
    id: str
    characters: list[Character]


@app.post("/api/rig")
def rig(req: RigRequest):
    image = get_scan(req.id)
    rigs, masks = [], []
    for c in req.characters:
        try:
            r, m = build_rig(image, c.model_dump())
        except (ValueError, RuntimeError):  # no usable mask / too few points to mesh
            continue
        r["texture"] = data_url(r.pop("rgba"), ".png")
        rigs.append(r)
        masks.append(m)
    if not rigs:
        raise HTTPException(422, "Couldn't cut the characters out of the picture")
    return {"background": data_url(fill_behind(image, masks), ".jpg"), "characters": rigs}


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/", StaticFiles(directory=WEB), name="web")
