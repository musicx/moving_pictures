"""Ask Claude what the drawing shows, where its joints are, and how it should move."""

import base64
import json
import os

import anthropic
import cv2
import numpy as np

MODEL = "claude-opus-5-5"
SEND_SIDE = 1024  # long side of the image sent to Claude; coordinates come back in this space

PROMPT = """这是一张小朋友美术作品的照片（画、拼贴、手工都有可能；照片里可能还拍到了桌子、手、颜料等）。图片尺寸 {w}x{h} 像素，坐标原点在左上角，x 向右、y 向下。
我们要让作品里的角色动起来，做成 {min_d}-{max_d} 秒、首尾无缝循环的 GIF。动画方式是：把角色从背景里抠出来，在角色内部放 2D 骨骼，绕关节旋转来变形原作（不重画、不加东西），角色原来位置的背景会被自动补上。

请完成：
0. artwork_box：作品本身的范围（纸/画板/手工作品的边缘），排除桌面、手、工具等杂物。
1. 找出作品里最适合动起来的角色（人和动物优先；如果没有，选一个最有生命感的东西，比如花、太阳、车）。最多 {max_chars} 个，按主次排序。地面线、背景装饰不要选。
2. 给每个角色一个紧贴它的 bbox（包住整个角色，包括翅膀尖、尾巴尖、头饰，但尽量不包进别的东西），以及一棵关节树：
   - 根关节放在躯干中心（parent 为空字符串）。
   - 其他关节放在**自然的转动点**：头颈连接处、肩、肘、手、髋、膝、脚、尾巴根部、尾巴中段和尖端、翅膀根部和尖端。
   - 每个关节、以及相邻关节之间的连线，都必须落在角色身体内部（不能落到背景上），这些位置会被当作"一定属于角色"来抠图。孩子画的比例可能很怪，按作品上实际的样子放，不要按真实解剖。
   - 一根"骨头"= 从 parent 关节到这个关节的线段。想让某部分单独转，它就需要从附着点出发的一根骨头，例如 body→tail_base→tail_tip，转 tail_tip 这根骨头就是绕 tail_base 摆尾巴。
   - 关节数量 4-14 个，只放会用到的。
3. 给出 3 个不同风格的动作方案（例如：摇尾巴、开心跳跳、挥手打招呼），每个方案对所有角色给出参数。动作要符合这个角色：狗摇尾巴、鸟扇翅膀、鱼摆尾、人挥手或跳舞。优先让最有表现力的部位动。
   - bones 里的 joint 指"以该关节为终点的那根骨头"，它绕其 parent 关节转动，子孙骨头跟着一起转。
   - amp_deg：摆幅（度）。四肢不超过 30，尾巴/翅膀不超过 40，头不超过 12。太大会把画撕坏。
   - cycles：在一个循环里摆几次，必须是正整数（1-4），这样首尾才能衔接。
   - phase：0-1，用来错开左右腿、前后腿等。
   - 全身动作：hop（跳的高度，占角色高度的比例，0-0.2）、float（上下漂浮幅度，0-0.1）、sway_deg（整体左右摇摆角度，0-8）、breathe（呼吸般的伸缩，0-0.08），全身动作的 cycles 也必须是正整数。不需要的设为 0。
   - 角色整体移动（hop、float）会露出它身后被自动修补的背景，背景越花哨越容易看出来。背景复杂时优先用局部摆动，hop 不超过 0.06。
   - duration_s：循环时长 {min_d}-{max_d} 秒。
   - label_zh 用小朋友能懂的 2-6 个字，emoji 一个。
4. title_zh：给这幅画起一个可爱的名字。narration_zh：用温柔、惊喜的语气对小朋友说一两句话，说出你看到了什么（会被念出来，40 字以内）。
"""


def _num():
    return {"type": "number"}


def _int():
    return {"type": "integer"}


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


SCHEMA = _obj({
    "title_zh": {"type": "string"},
    "artwork_box": _obj({"x0": _int(), "y0": _int(), "x1": _int(), "y1": _int()}),
    "narration_zh": {"type": "string"},
    "characters": {"type": "array", "items": _obj({
        "id": {"type": "string"},
        "name_zh": {"type": "string"},
        "kind": {"type": "string", "enum": ["human", "quadruped", "bird", "fish", "insect", "other"]},
        "bbox": _obj({"x0": _int(), "y0": _int(), "x1": _int(), "y1": _int()}),
        "joints": {"type": "array", "items": _obj({
            "name": {"type": "string"}, "x": _int(), "y": _int(), "parent": {"type": "string"},
        })},
    })},
    "motions": {"type": "array", "items": _obj({
        "label_zh": {"type": "string"},
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
        raise RuntimeError("Claude 没有接受这张图片")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Claude 的回答被截断了")
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
        chars.append({"id": c["id"], "name_zh": c["name_zh"], "kind": c["kind"], "bbox": bbox, "joints": joints})

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
            motions.append({"label_zh": m["label_zh"], "emoji": m["emoji"],
                            "duration_s": _clamp(m["duration_s"], MIN_D, MAX_D), "tracks": tracks})
    if not chars or not motions:
        raise RuntimeError("没能从画里找到可以动起来的角色")
    return {"title_zh": raw["title_zh"], "artwork_box": artwork, "narration_zh": raw["narration_zh"], "characters": chars, "motions": motions}


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
