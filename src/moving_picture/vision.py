"""Photo of an artwork -> straightened image, and per-character masks cut out of a busy background."""

import io

import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()  # iPhone photos arrive as HEIC

MAX_SIDE = 1600
GRABCUT_SIDE = 640


def load_photo(data: bytes) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return np.ascontiguousarray(np.asarray(img))


def find_sheet(rgb: np.ndarray) -> np.ndarray | None:
    """Corners of a rectangular sheet lying on a darker surface, or None.

    Only a clean quadrilateral is accepted; round boards, hands over the paper or a sheet
    filling the whole frame return None and the photo is used as is."""
    h, w = rgb.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (0, 0), 5)
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if not 0.25 * h * w < hull_area < 0.92 * h * w or cv2.contourArea(cnt) < 0.9 * hull_area:
        return None
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.05, 9):
        quad = cv2.approxPolyDP(hull, eps * peri, True)
        if len(quad) == 4:
            if abs(cv2.contourArea(quad) / hull_area - 1) > 0.04:
                return None
            return _order_corners(quad.reshape(4, 2).astype(np.float32))
    return None


def _order_corners(pts: np.ndarray) -> np.ndarray:
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]])


def straighten(rgb: np.ndarray) -> np.ndarray:
    quad = find_sheet(rgb)
    if quad is None:
        return rgb
    tl, tr, br, bl = quad
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    return cv2.warpPerspective(rgb, cv2.getPerspectiveTransform(quad, dst), (w, h), flags=cv2.INTER_CUBIC)


def _fill_holes(m: np.ndarray) -> np.ndarray:
    padded = cv2.copyMakeBorder(m.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(padded, None, (0, 0), 2)
    return (padded != 2)[1:-1, 1:-1]


def character_mask(image: np.ndarray, bbox: list[int], joints: list[dict]) -> np.ndarray:
    """Cut one character out with GrabCut.

    Claude's bbox says roughly where it is; its bones run through the body, so they seed
    'definitely character'. Outside a margin around the bbox is 'definitely background'."""
    H, W = image.shape[:2]
    x0, y0, x1, y1 = bbox
    mx, my = int(0.15 * (x1 - x0)) + 4, int(0.15 * (y1 - y0)) + 4
    rx0, ry0, rx1, ry1 = max(0, x0 - mx), max(0, y0 - my), min(W, x1 + mx), min(H, y1 + my)
    region = image[ry0:ry1, rx0:rx1]
    k = min(1.0, GRABCUT_SIDE / max(region.shape[:2]))
    small = cv2.resize(region, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else region.copy()
    sh, sw = small.shape[:2]

    gc = np.full((sh, sw), cv2.GC_PR_BGD, np.uint8)
    border = max(2, int(0.02 * max(sh, sw)))
    gc[:border], gc[-border:], gc[:, :border], gc[:, -border:] = (cv2.GC_BGD,) * 4
    bx0, by0 = int((x0 - rx0) * k), int((y0 - ry0) * k)
    bx1, by1 = int((x1 - rx0) * k), int((y1 - ry0) * k)
    gc[by0:by1, bx0:bx1] = cv2.GC_PR_FGD
    by_name = {j["name"]: j for j in joints}
    pt = lambda j: (int((j["x"] - rx0) * k), int((j["y"] - ry0) * k))
    seeds = np.zeros((sh, sw), np.uint8)
    thick = max(2, int(0.012 * max(sh, sw)))
    for j in joints:
        if j["parent"] in by_name:
            cv2.line(seeds, pt(by_name[j["parent"]]), pt(j), 1, thick)
        cv2.circle(seeds, pt(j), thick, 1, -1)
    gc[seeds > 0] = cv2.GC_FGD

    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, 6, cv2.GC_INIT_WITH_MASK)
    fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)

    # Keep only pieces connected to the skeleton, then tidy the edge.
    n, labels = cv2.connectedComponents(fg)
    hit = np.unique(labels[(seeds > 0) & (fg > 0)])
    fg = np.isin(labels, hit[hit > 0])
    r = max(1, int(0.006 * max(sh, sw)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, ker)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, ker)
    fg = _fill_holes(fg)

    full = cv2.resize(fg.astype(np.uint8) * 255, (rx1 - rx0, ry1 - ry0), interpolation=cv2.INTER_LINEAR) > 127
    mask = np.zeros((H, W), bool)
    mask[ry0:ry1, rx0:rx1] = full
    return mask


def fill_behind(image: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    """Paint the background where characters used to be, so moving them reveals no hole."""
    if not masks:
        return image
    m = cv2.dilate(np.any(masks, axis=0).astype(np.uint8), np.ones((9, 9), np.uint8))
    h, w = m.shape
    k = min(1.0, 600 / max(h, w))  # inpainting is slow at full size; the filled area is soft anyway
    small = cv2.resize(image, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    sm = cv2.resize(m, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    filled = cv2.inpaint(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), sm, 7, cv2.INPAINT_TELEA)
    filled = cv2.cvtColor(cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    out = image.copy()
    out[m > 0] = filled[m > 0]
    return out


def outline_points(mask: np.ndarray, n: int = 300) -> list[list[int]]:
    """Evenly spaced points along a mask's outline, for the sparkle trace."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = [c.reshape(-1, 2) for c in contours if len(c) > 20]
    if not pts:
        return []
    pts = np.concatenate(pts)
    idx = np.linspace(0, len(pts) - 1, min(n, len(pts))).astype(int)
    return pts[idx].tolist()
