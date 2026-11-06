"""Synthetic 'phone photo of a child's drawing' for tests and local demos.

Run `uv run python tests/synth.py out_dir/` to write sample JPEGs."""

import sys
from pathlib import Path

import cv2
import numpy as np

PAPER_W, PAPER_H = 1100, 800


def _wobbly(pts, rng, amp=3.0):
    pts = np.asarray(pts, np.float64)
    return (pts + rng.normal(0, amp, pts.shape)).astype(np.int32)


def _ellipse_pts(cx, cy, rx, ry, n=60):
    t = np.linspace(0, 2 * np.pi, n)
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1)


def crayon_fill(img, poly, color, rng, density=0.55):
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, [poly], 1)
    speck = rng.random(img.shape[:2]) < density
    sel = (mask > 0) & speck
    img[sel] = color


def draw_cat(img, rng, ox=300, oy=300, line=(40, 40, 40)):
    """A side-view cat: body, head, ears, four legs, a tail. Returns ground-truth joints."""
    body = _wobbly(_ellipse_pts(ox + 200, oy + 160, 170, 85), rng)
    head = _wobbly(_ellipse_pts(ox + 400, oy + 60, 75, 65), rng)
    crayon_fill(img, body, (60, 140, 235), rng)  # orange crayon (BGR)
    crayon_fill(img, head, (60, 140, 235), rng)
    for p in (body, head):
        cv2.polylines(img, [p], True, line, 5, cv2.LINE_AA)
    for ear in ([(ox + 345, oy + 15), (ox + 360, oy - 50), (ox + 395, oy)], [(ox + 410, oy - 2), (ox + 450, oy - 45), (ox + 460, oy + 20)]):
        cv2.polylines(img, [_wobbly(ear, rng, 2)], True, line, 5, cv2.LINE_AA)
    for lx in (ox + 90, ox + 150, ox + 260, ox + 320):
        leg = _wobbly([(lx, oy + 220), (lx - 5, oy + 330), (lx + 30, oy + 335), (lx + 30, oy + 225)], rng, 2)
        cv2.fillPoly(img, [leg], (60, 140, 235))
        cv2.polylines(img, [leg], False, line, 5, cv2.LINE_AA)
    t = np.linspace(0, 1, 30)
    tail = np.stack([ox + 40 - 120 * t, oy + 140 - 200 * t + 80 * t**2], 1)
    cv2.polylines(img, [_wobbly(tail, rng, 1.5)], False, line, 14, cv2.LINE_AA)
    cv2.circle(img, (ox + 420, oy + 50), 8, (20, 20, 20), -1)  # eye
    return {
        "body": (ox + 200, oy + 160),
        "tail_base": (ox + 40, oy + 140),
        "tail_tip": (ox - 80, oy + 20),
        "neck": (ox + 340, oy + 110),
        "head": (ox + 410, oy + 50),
    }


def draw_sun(img, rng, cx=950, cy=120):
    cv2.circle(img, (cx, cy), 55, (40, 200, 250), -1)
    for a in np.linspace(0, 2 * np.pi, 10, endpoint=False):
        p0 = (int(cx + 70 * np.cos(a)), int(cy + 70 * np.sin(a)))
        p1 = (int(cx + 110 * np.cos(a)), int(cy + 110 * np.sin(a)))
        cv2.line(img, p0, p1, (40, 200, 250), 6)


def make_photo(seed=0, with_table=True):
    rng = np.random.default_rng(seed)
    paper = np.full((PAPER_H, PAPER_W, 3), (238, 242, 245), np.uint8)
    paper = (paper + rng.normal(0, 3, paper.shape)).clip(0, 255).astype(np.uint8)
    joints = draw_cat(paper, rng)
    draw_sun(paper, rng)
    cv2.line(paper, (60, 700), (1040, 690), (60, 160, 60), 6)  # grass line
    if not with_table:
        return paper, joints
    H, W = 1200, 1600
    table = np.zeros((H, W, 3), np.uint8)
    table[:] = (70, 100, 140)
    table = (table + rng.normal(0, 6, table.shape)).clip(0, 255).astype(np.uint8)
    src = np.float32([[0, 0], [PAPER_W, 0], [PAPER_W, PAPER_H], [0, PAPER_H]])
    dst = np.float32([[260, 200], [1380, 260], [1320, 1050], [200, 960]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(paper, M, (W, H))
    inside = cv2.warpPerspective(np.ones((PAPER_H, PAPER_W), np.uint8), M, (W, H)).astype(bool)
    table[inside] = warped[inside]
    yy, xx = np.mgrid[0:H, 0:W]
    light = 0.65 + 0.35 * (xx / W) * (1 - 0.4 * yy / H)  # uneven light, darker bottom-left
    photo = (table * light[..., None]).clip(0, 255).astype(np.uint8)
    return photo, joints


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    photo, _ = make_photo()
    cv2.imwrite(str(out / "synthetic_cat_photo.jpg"), photo)
    print(out / "synthetic_cat_photo.jpg")
