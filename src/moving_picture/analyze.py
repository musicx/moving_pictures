"""Ask Claude what the drawing shows, where its joints are, and how it should move."""

import base64
import json
import os

import anthropic
import cv2
import numpy as np

MODEL = "claude-opus-5-5"
SEND_SIDE = 1024  # long side of the image sent to Claude; coordinates come back in this space

PROMPT = """This is a photo of a child's artwork (drawing, painting, collage or craft; the photo may also show the table, hands, paints, etc.). The image is {w}x{h} pixels, origin at the top-left, x to the right, y down.
We will bring the characters in the artwork to life as a {min_d}-{max_d} second GIF that loops seamlessly. Each character is cut out of the background, a 2D skeleton is placed inside it, and the original artwork is deformed by rotating around the joints (nothing is redrawn or added). The background behind the character is filled in automatically.

Please do the following:
0. artwork_box: the extent of the artwork itself (edges of the paper, board or craft), excluding the table, hands, tools and other clutter.
1. Find the characters best suited to move (people and animals first; if there are none, pick the most lively thing, such as a flower, the sun or a car). At most {max_chars}, most important first. Do not pick ground lines or background decoration.
2. For each character give a tight bbox (the whole character including wing tips, tail tips and headwear, but as little else as possible), and a joint tree:
   - The root joint sits at the center of the torso (parent is the empty string).
   - Other joints sit at **natural pivot points**: base of the neck, shoulders, elbows, hands, hips, knees, feet, base/middle/tip of the tail, base and tip of wings.
   - Every joint, and the line between each joint and its parent, must lie inside the character's body (never on the background); these are used as "definitely the character" when cutting it out. Children draw odd proportions: place joints where the parts actually are in the artwork, not where anatomy says.
   - A "bone" is the segment from a parent joint to a joint. For a part to move on its own it needs a bone starting at its attachment point, e.g. body -> tail_base -> tail_tip: rotating the tail_tip bone swings the tail around tail_base.
   - Use 4-14 joints, only the ones the motions need.
3. Give 3 motions with distinct moods (e.g. wag the tail, happy hops, wave hello); each motion gives parameters for every character. Motions must suit the character: a dog wags its tail, a bird flaps its wings, a fish swishes its tail, a person waves or dances. Prefer moving the most expressive part.
   - joint in bones means "the bone ending at this joint"; it rotates around its parent joint and its descendants follow.
   - amp_deg: swing amplitude in degrees. Limbs at most 30, tails/wings at most 40, heads at most 12. Larger values tear the artwork.
   - cycles: swings per loop, a positive integer (1-4), so the loop joins up seamlessly.
   - phase: 0-1, to offset left/right or front/back legs.
   - Whole-body motion: hop (jump height as a fraction of the character's height, 0-0.2), float (up-and-down drift, 0-0.1), sway_deg (whole-body side-to-side rock, 0-8), breathe (breathing stretch, 0-0.08); the whole-body cycles must also be a positive integer. Set unused ones to 0.
   - Moving the whole character (hop, float) reveals the filled-in background behind it, which is more visible on busy backgrounds. On busy backgrounds prefer moving parts, and keep hop at most 0.06.
   - duration_s: loop length, {min_d}-{max_d} seconds.
   - label: 1-3 simple words a young child understands, without emoji; emoji: exactly one emoji for the motion.
   - video_prompt: a prompt for an image-to-video model (Grok Imagine) whose first frame is this photo cropped to artwork_box. Describe about 5 seconds of this motion in simple, vivid words: who moves and how, while everything else stays still. Ask it to keep the exact look of the child's artwork (same colors, paper/collage/paint texture, hand-drawn lines), a static camera, no new objects or characters, and no text. Under 60 words.
4. title: a cute name for the artwork. narration: one or two warm, delighted sentences to the child saying what you see (it will be read aloud; under 25 words). Write all text in English.
"""


def _num():
    return {"type": "number"}


def _int():
    return {"type": "integer"}


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


SCHEMA = _obj({
    "title": {"type": "string"},
    "artwork_box": _obj({"x0": _int(), "y0": _int(), "x1": _int(), "y1": _int()}),
    "narration": {"type": "string"},
    "characters": {"type": "array", "items": _obj({
        "id": {"type": "string"},
        "name": {"type": "string"},
        "kind": {"type": "string", "enum": ["human", "quadruped", "bird", "fish", "insect", "other"]},
        "bbox": _obj({"x0": _int(), "y0": _int(), "x1": _int(), "y1": _int()}),
        "joints": {"type": "array", "items": _obj({
            "name": {"type": "string"}, "x": _int(), "y": _int(), "parent": {"type": "string"},
        })},
    })},
    "motions": {"type": "array", "items": _obj({
        "label": {"type": "string"},
        "video_prompt": {"type": "string"},
        "emoji": {"type": "string"},
        "duration_s": _num(),
        "tracks": {"type": "array", "items": _obj({
            "character_id": {"type": "string"},
            "hop": _num(), "float": _num(), "sway_deg": _num(), "breathe": _num(), "cycles": _int(),
            "bones": {"type": "array", "items": _obj({
                "joint": {"type": "string"}, "amp_deg": _num(), "cycles": _int(), "phase": _num(),
            })},
        })},
    })},
})

LIMITS = {"hop": 0.2, "float": 0.1, "sway_deg": 8, "breathe": 0.08}
MIN_D, MAX_D, MAX_CHARS = 1.0, 3.0, 3


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def ask_claude(image: np.ndarray) -> dict:
    h, w = image.shape[:2]
    scale = SEND_SIDE / max(h, w)
    small = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    sh, sw = small.shape[:2]
    prompt = PROMPT.format(w=sw, h=sh, min_d=MIN_D, max_d=MAX_D, max_chars=MAX_CHARS)

    # The project's .env names the key CLAUDE_API_KEY; None falls back to the SDK's usual lookup.
    client = anthropic.Anthropic(api_key=os.environ.get("CLAUDE_API_KEY"))
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.standard_b64encode(jpg.tobytes()).decode()}},
            {"type": "text", "text": prompt},
        ]}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined this picture")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Claude's answer was cut off")
    text = next(b.text for b in response.content if b.type == "text")
    return sanitize(json.loads(text), 1 / scale, w, h)


def sanitize(raw: dict, k: float, w: int, h: int) -> dict:
    """Scale Claude's coordinates back to the scan, and keep only a well-formed rig and motion."""
    def box(b):
        return [_clamp(round(b["x0"] * k), 0, w), _clamp(round(b["y0"] * k), 0, h),
                _clamp(round(b["x1"] * k), 0, w), _clamp(round(b["y1"] * k), 0, h)]

    artwork = box(raw["artwork_box"])
    if artwork[2] - artwork[0] < 0.2 * w or artwork[3] - artwork[1] < 0.2 * h:
        artwork = [0, 0, w, h]
    chars = []
    for c in raw["characters"][:MAX_CHARS]:
        bbox = box(c["bbox"])
        if bbox[2] - bbox[0] < 8 or bbox[3] - bbox[1] < 8:
            continue
        joints = {}
        for j in c["joints"]:
            if j["name"] in joints:
                continue
            joints[j["name"]] = {"name": j["name"], "x": _clamp(round(j["x"] * k), 0, w - 1),
                                 "y": _clamp(round(j["y"] * k), 0, h - 1), "parent": j["parent"] or ""}
        joints = _as_tree(list(joints.values()))
        if not joints:
            continue
        chars.append({"id": c["id"], "name": c["name"], "kind": c["kind"], "bbox": bbox, "joints": joints})

    by_id = {c["id"]: {j["name"] for j in c["joints"] if j["parent"]} for c in chars}
    motions = []
    for m in raw["motions"]:
        tracks = []
        for t in m["tracks"]:
            bones_ok = by_id.get(t["character_id"])
            if bones_ok is None:
                continue
            track = {key: _clamp(abs(t[key]), 0, lim) for key, lim in LIMITS.items()}
            track["character_id"] = t["character_id"]
            track["cycles"] = _clamp(int(t["cycles"]), 1, 4)
            track["bones"] = [
                {"joint": bn["joint"], "amp_deg": _clamp(bn["amp_deg"], -40, 40),
                 "cycles": _clamp(int(bn["cycles"]), 1, 4), "phase": bn["phase"] % 1.0}
                for bn in t["bones"] if bn["joint"] in bones_ok
            ]
            tracks.append(track)
        if tracks:
            motions.append({"label": m["label"], "emoji": m["emoji"], "video_prompt": m["video_prompt"],
                            "duration_s": _clamp(m["duration_s"], MIN_D, MAX_D), "tracks": tracks})
    if not chars or not motions:
        raise RuntimeError("Couldn't find anyone in the picture who can move")
    return {"title": raw["title"], "artwork_box": artwork, "narration": raw["narration"], "characters": chars, "motions": motions}


def _as_tree(joints: list[dict]) -> list[dict]:
    """Keep the joints reachable from a single root; parents always come before children."""
    names = {j["name"] for j in joints}
    for j in joints:
        if j["parent"] not in names:
            j["parent"] = ""
    roots = [j for j in joints if not j["parent"]]
    if not roots:
        return []
    root = roots[0]
    for r in roots[1:]:  # stray roots hang off the main one
        r["parent"] = root["name"]
    children: dict[str, list[dict]] = {}
    for j in joints:
        if j["parent"]:
            children.setdefault(j["parent"], []).append(j)
    ordered, stack = [], [root]
    while stack:
        j = stack.pop()
        ordered.append(j)
        stack.extend(children.get(j["name"], []))
    return ordered
