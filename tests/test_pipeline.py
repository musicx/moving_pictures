import cv2
import numpy as np
import pytest

from moving_picture.analyze import sanitize
from moving_picture.rig import build_rig
from moving_picture.vision import character_mask, find_sheet, straighten
from synth import PAPER_H, PAPER_W, make_photo


@pytest.fixture(scope="module")
def paper():
    img, joints = make_photo(with_table=False)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), joints


def cat_character(j):
    names = [("body", ""), ("tail_base", "body"), ("tail_tip", "tail_base"), ("neck", "body"), ("head", "neck")]
    return {"id": "cat", "name_zh": "猫", "bbox": [200, 240, 800, 650],
            "joints": [{"name": n, "x": j[n][0], "y": j[n][1], "parent": p} for n, p in names]}


def test_straighten_finds_the_sheet_on_a_table():
    photo, _ = make_photo()
    rgb = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
    assert find_sheet(rgb) is not None
    h, w = straighten(rgb).shape[:2]
    assert abs(w / h - PAPER_W / PAPER_H) < 0.05


def test_sheet_filling_the_frame_is_left_alone(paper):
    assert find_sheet(paper[0]) is None


def test_character_mask_cuts_out_the_cat_only(paper):
    img, j = paper
    m = character_mask(img, cat_character(j)["bbox"], cat_character(j)["joints"])
    assert m[j["body"][1], j["body"][0]] and m[j["head"][1], j["head"][0]]
    assert m[j["tail_tip"][1] + 5, j["tail_tip"][0] + 5] or m[j["tail_tip"][1], j["tail_tip"][0]]
    assert not m[120, 950]  # sun
    assert not m[695, 500]  # grass line
    assert not m[600, 515]  # paper between the front and back legs


def test_rig_weights_follow_the_bones(paper):
    img, j = paper
    rig, _ = build_rig(img, cat_character(j))
    v = np.array(rig["vertices"]).reshape(-1, 2)
    inf = np.array(rig["influences"]).reshape(len(v), -1, 2)
    assert np.allclose(inf[:, :, 1].sum(1), 1, atol=1e-3)
    assert np.array(rig["triangles"]).max() < len(v)
    bones = [b["joint"] for b in rig["bones"]]
    tip = np.argmin(np.hypot(*(v - j["tail_tip"]).T))
    head = np.argmin(np.hypot(*(v - j["head"]).T))
    dominant = lambda i: bones[int(inf[i, inf[i, :, 1].argmax(), 0])]
    assert dominant(tip) == "tail_tip"
    assert dominant(head) == "head"


def test_sanitize_scales_clamps_and_repairs():
    raw = {
        "title_zh": "t", "narration_zh": "n",
        "artwork_box": {"x0": 0, "y0": 0, "x1": 5, "y1": 5},  # too small -> whole image
        "characters": [{"id": "a", "name_zh": "a", "kind": "quadruped", "bbox": {"x0": 10, "y0": 10, "x1": 60, "y1": 60},
                        "joints": [{"name": "body", "x": 30, "y": 30, "parent": ""},
                                   {"name": "tail", "x": 50, "y": 40, "parent": "body"},
                                   {"name": "paw", "x": 20, "y": 55, "parent": "missing"}]}],
        "motions": [{"label_zh": "摇", "emoji": "x", "duration_s": 9,
                     "tracks": [{"character_id": "a", "hop": 1, "float": -0.5, "sway_deg": 3, "breathe": 0, "cycles": 0,
                                 "bones": [{"joint": "tail", "amp_deg": 90, "cycles": 7, "phase": 1.25},
                                           {"joint": "body", "amp_deg": 10, "cycles": 1, "phase": 0},
                                           {"joint": "nope", "amp_deg": 10, "cycles": 1, "phase": 0}]},
                                {"character_id": "ghost", "hop": 0, "float": 0, "sway_deg": 0, "breathe": 0, "cycles": 1, "bones": []}]}],
    }
    out = sanitize(raw, 2.0, 200, 200)
    assert out["artwork_box"] == [0, 0, 200, 200]
    c = out["characters"][0]
    assert c["bbox"] == [20, 20, 120, 120]
    assert {j["name"]: j["parent"] for j in c["joints"]} == {"body": "", "tail": "body", "paw": "body"}
    m = out["motions"][0]
    assert m["duration_s"] == 3.0
    assert len(m["tracks"]) == 1
    t = m["tracks"][0]
    assert (t["hop"], t["float"], t["cycles"]) == (0.2, 0.1, 1)
    assert t["bones"] == [{"joint": "tail", "amp_deg": 40, "cycles": 4, "phase": 0.25}]
