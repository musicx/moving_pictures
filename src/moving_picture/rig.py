"""Turn a character on the scan into a skinned triangle mesh the browser can deform."""

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import Delaunay
from skimage.graph import MCP_Geometric

from .vision import character_mask, outline_points

WEIGHT_SIDE = 200  # geodesic weights are computed on a mask downscaled to this long side
MAX_INFLUENCES = 4


def _snap(joints: list[dict], mask: np.ndarray) -> list[dict]:
    _, (iy, ix) = distance_transform_edt(~mask, return_indices=True)
    out = []
    for j in joints:
        x, y = j["x"], j["y"]
        out.append({**j, "x": int(ix[y, x]), "y": int(iy[y, x])})
    return out


def _mesh(mask: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    m8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = []
    for c in contours:
        c = c.reshape(-1, 2)
        if len(c) < 3:
            continue
        pts.append(c[:: max(1, int(step / 1.5))])
    inner = distance_transform_edt(mask)
    ys, xs = np.mgrid[step / 2 : mask.shape[0] : step, step / 2 : mask.shape[1] : step]
    ys, xs = ys.ravel().astype(int), xs.ravel().astype(int)
    ok = inner[ys, xs] > step * 0.5
    pts.append(np.stack([xs[ok], ys[ok]], 1))
    verts = np.unique(np.concatenate(pts), axis=0).astype(np.float64)
    tri = Delaunay(verts).simplices
    # Keep triangles that lie on the drawing; drop ones bridging gaps (e.g. between two legs).
    tol = cv2.dilate(m8, np.ones((3, 3), np.uint8)).astype(bool)
    a, b, c = verts[tri[:, 0]], verts[tri[:, 1]], verts[tri[:, 2]]
    ok = np.ones(len(tri), bool)
    for p in ((a + b + c) / 3, (a + b) / 2, (b + c) / 2, (c + a) / 2, (2 * a + b) / 3, (2 * b + c) / 3, (2 * c + a) / 3):
        pi = np.round(p).astype(int)
        ok &= tol[pi[:, 1], pi[:, 0]]
    u, v = b - a, c - a
    area = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
    ok &= area > 0.5
    tri = tri[ok]
    used = np.unique(tri)
    remap = -np.ones(len(verts), int)
    remap[used] = np.arange(len(used))
    return verts[used], remap[tri]


def _segment_pixels(p, c, n=40, start=0.3):
    t = np.linspace(start, 1, n)[:, None]
    return np.asarray(p) * (1 - t) + np.asarray(c) * t


def _weights(mask: np.ndarray, verts: np.ndarray, bones: list[dict]) -> np.ndarray:
    """Geodesic (inside-the-drawing) distance from each vertex to each bone -> skinning weights.

    Bone 0 is the rigid root around the root joint; it keeps the torso from being dragged
    by limbs that hang off the root."""
    f = WEIGHT_SIDE / max(mask.shape)
    small = cv2.resize(mask.astype(np.uint8), (max(1, round(mask.shape[1] * f)), max(1, round(mask.shape[0] * f))),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    if not small.any():
        small = cv2.resize(mask.astype(np.uint8), small.shape[::-1], interpolation=cv2.INTER_AREA) > 0
    costs = np.where(small, 1.0, np.inf)
    _, (iy, ix) = distance_transform_edt(~small, return_indices=True)
    sh, sw = small.shape

    def to_small(pts):
        q = np.clip(np.round(np.asarray(pts) * f).astype(int), 0, [sw - 1, sh - 1])
        return iy[q[:, 1], q[:, 0]], ix[q[:, 1], q[:, 0]]  # nearest pixel inside the mask

    vy, vx = to_small(verts)
    dist = np.empty((len(verts), len(bones)))
    for k, bone in enumerate(bones):
        if bone["parent"] is None:
            seg = np.array([[bone["cx"], bone["cy"]]])
        else:
            seg = _segment_pixels((bone["px"], bone["py"]), (bone["cx"], bone["cy"]))
        sy, sx = to_small(seg)
        cum, _ = MCP_Geometric(costs).find_costs(list(zip(sy, sx)))
        d = cum[vy, vx]
        dist[:, k] = np.where(np.isfinite(d), d, 1e6)
    w = 1.0 / (dist + 2.0) ** 4
    top = np.argsort(-w, axis=1)[:, :MAX_INFLUENCES]
    tw = np.take_along_axis(w, top, 1)
    tw /= tw.sum(1, keepdims=True)
    return np.stack([top, tw], 2)  # (n, MAX_INFLUENCES, [bone index, weight])


def build_rig(image: np.ndarray, character: dict) -> tuple[dict, np.ndarray]:
    """Returns the rig for the browser and the character's mask (to paint over in the background)."""
    mask = character_mask(image, character["bbox"], character["joints"])
    if not mask.any():
        raise ValueError(f"角色 {character['name_zh']} 在画里找不到")
    joints = _snap(character["joints"], mask)

    ys, xs = np.nonzero(mask)
    pad = 6
    h, w = mask.shape
    x0, y0 = int(max(0, xs.min() - pad)), int(max(0, ys.min() - pad))
    x1, y1 = int(min(w, xs.max() + pad + 1)), int(min(h, ys.max() + pad + 1))
    crop = mask[y0:y1, x0:x1]
    # Mesh covers a slightly grown mask so anti-aliased stroke edges are not cut off.
    grown = cv2.dilate(crop.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).astype(bool)
    step = max(6.0, max(crop.shape) / 28)
    verts, tris = _mesh(grown, step)

    by_name = {j["name"]: j for j in joints}
    root = joints[0]
    bones = [{"joint": root["name"], "parent": None, "px": root["x"] - x0, "py": root["y"] - y0,
              "cx": root["x"] - x0, "cy": root["y"] - y0}]
    for j in joints[1:]:
        p = by_name[j["parent"]]
        bones.append({"joint": j["name"], "parent": p["name"], "px": p["x"] - x0, "py": p["y"] - y0,
                      "cx": j["x"] - x0, "cy": j["y"] - y0})
    weights = _weights(grown, verts, bones)

    alpha = cv2.GaussianBlur(grown.astype(np.float32), (0, 0), 0.8)
    rgba = np.dstack([image[y0:y1, x0:x1], (alpha * 255).astype(np.uint8)])
    for b in bones:  # browser works in scan coordinates
        b["px"] += x0
        b["py"] += y0
        b["cx"] += x0
        b["cy"] += y0
    rig = {
        "id": character["id"],
        "x": int(x0), "y": int(y0), "w": int(x1 - x0), "h": int(y1 - y0),
        "rgba": rgba,
        "vertices": (verts + [x0, y0]).round(1).ravel().tolist(),
        "triangles": tris.ravel().tolist(),
        "bones": bones,
        "influences": [round(float(v), 4) for v in weights.ravel()],
        "joints": joints,
        "bbox": character["bbox"],
        "outline": outline_points(mask),
    }
    return rig, mask
