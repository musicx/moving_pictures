import { Renderer, buildScene, exportGif } from "./puppet.js";
import { Sparkles, speak, starrySky } from "./magic.js";

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

starrySky($("sky"));
const sparkles = new Sparkles($("sparkles"));
const renderer = new Renderer($("gl"));

const WAIT_LINES = ["✨ 魔法正在认识你的画…", "🔍 看看都有谁…", "🎨 这些颜色真漂亮…", "🦶 数一数有几只小脚…", "💫 找一找谁最想动起来…"];

let state = null; // { scan, analysis, view, scene, motion, t0, aliveAt, editing }

function show(stage) {
  for (const id of ["home", "camera", "show"]) $(id).hidden = id !== stage;
}

function say(text, { pop = false, error = false } = {}) {
  const m = $("message");
  m.textContent = text;
  m.classList.toggle("error", error);
  m.classList.remove("pop");
  if (pop) void m.offsetWidth, m.classList.add("pop");
}

async function api(path, body) {
  const res = await fetch(path, body instanceof FormData ? { method: "POST", body } : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || "魔法出了点小问题");
  return data;
}

// Size the picture box to the view (a rectangle of the photo) and place the photo inside it.
function layout(view, animate) {
  const [x0, y0, x1, y1] = view, vw = x1 - x0, vh = y1 - y0;
  const frame = $("frame").getBoundingClientRect();
  const maxW = frame.width - 36, maxH = window.innerHeight * 0.62;
  const s = Math.min(maxW / vw, maxH / vh);
  const box = $("canvas-box"), img = $("photo");
  box.style.transition = img.style.transition = animate ? "all 1.1s cubic-bezier(.6,0,.2,1)" : "none";
  Object.assign(box.style, { width: `${vw * s}px`, height: `${vh * s}px` });
  Object.assign(img.style, {
    width: `${state.scan.width * s}px`, height: `${state.scan.height * s}px`, left: `${-x0 * s}px`, top: `${-y0 * s}px`,
  });
  state.scale = s;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  for (const c of [$("gl"), $("joints-layer")]) {
    c.width = Math.round(vw * s * dpr);
    c.height = Math.round(vh * s * dpr);
  }
}

const toBox = ([x, y]) => [(x - state.view[0]) * state.scale, (y - state.view[1]) * state.scale];

async function start(file) {
  if (!file) return;
  stopAnimation();
  $("controls").hidden = true;
  $("gl").classList.remove("on");
  $("photo").style.opacity = 1;
  show("show");
  say("🪄 魔法正在打开你的画…");
  try {
    const fd = new FormData();
    fd.append("file", file);
    const scan = await api("/api/scan", fd);
    state = { scan, view: [0, 0, scan.width, scan.height] };
    $("photo").src = scan.image;
    layout(state.view, false);
    await sleep(50);
    sparkles.resize();
    $("scanbar").classList.add("on");
    sparkles.glimmer(true);
    let i = 0;
    say(WAIT_LINES[0]);
    const ticker = setInterval(() => say(WAIT_LINES[++i % WAIT_LINES.length]), 3200);
    let analysis;
    try {
      analysis = await api("/api/analyze", { id: scan.id });
    } finally {
      clearInterval(ticker);
      $("scanbar").classList.remove("on");
      sparkles.glimmer(false);
    }
    state.analysis = analysis;
    state.view = analysis.artwork_box;

    const rigging = api("/api/rig", { id: scan.id, characters: analysis.characters });
    layout(state.view, true);
    $("frame").classList.add("glowing");
    say(`我看到了……${analysis.title_zh}！`, { pop: true });
    speak(analysis.narration_zh);
    await sleep(1200);
    sparkles.resize();

    const rig = await rigging;
    await loadRig(rig);
    for (const c of rig.characters) await sparkles.trace(c.outline.map(toBox), 1800);
    for (const c of rig.characters) {
      const [x, y] = toBox([c.x + c.w / 2, c.y + c.h / 2]);
      sparkles.burst(x, y);
    }
    $("gl").classList.add("on");
    await sleep(300);
    $("photo").style.opacity = 0;
    $("frame").classList.remove("glowing");
    say(analysis.narration_zh, { pop: true });
    showMotions();
    play(analysis.motions[0]);
  } catch (e) {
    console.error(e);
    stopAnimation();
    say(`😢 ${e.message}`, { error: true });
    $("controls").hidden = false;
    for (const id of ["motions", "btn-gif", "btn-edit"]) $(id).hidden = true;
  }
}

async function loadRig(rig) {
  state.scene = await buildScene(state.scan, state.view, rig);
  await renderer.setScene(state.scene);
  // Keep the joints the server snapped onto the character, so editing starts from them.
  for (const r of rig.characters) {
    const c = state.analysis.characters.find((c) => c.id === r.id);
    if (c) c.joints = r.joints;
  }
}

function showMotions() {
  const box = $("motions");
  box.innerHTML = "";
  state.analysis.motions.forEach((m, i) => {
    const b = document.createElement("button");
    b.className = "magic-btn";
    b.textContent = `${m.emoji} ${m.label_zh}`;
    b.onclick = () => play(m);
    b.dataset.i = i;
    box.appendChild(b);
  });
  $("controls").hidden = false;
  for (const id of ["motions", "btn-gif", "btn-edit"]) $(id).hidden = false;
}

let raf = 0;
function stopAnimation() {
  cancelAnimationFrame(raf);
  raf = 0;
}

function play(motion) {
  state.motion = motion;
  state.t0 = state.aliveAt = performance.now();
  [...$("motions").children].forEach((b) => b.classList.toggle("selected", state.analysis.motions[b.dataset.i] === motion));
  stopAnimation();
  const frame = (now) => {
    if (state.editing) {
      renderer.draw(null, 0);
    } else {
      const u = (((now - state.t0) / 1000) / motion.duration_s) % 1;
      const k = Math.min(1, (now - state.aliveAt) / 1500);
      renderer.draw(motion, u, k * k * (3 - 2 * k)); // ease into motion
    }
    raf = requestAnimationFrame(frame);
  };
  raf = requestAnimationFrame(frame);
}

async function saveGif() {
  const btn = $("btn-gif");
  btn.disabled = true;
  say("🖼️ 正在把魔法装进相框…");
  try {
    const blob = await exportGif(state.scene, state.motion);
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${state.analysis.title_zh}-${state.motion.label_zh}.gif`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
    say("🎉 保存好啦！", { pop: true });
  } catch (e) {
    console.error(e);
    say("😢 保存失败了，再试一次吧", { error: true });
  } finally {
    btn.disabled = false;
  }
}

// --- Joint editing (for grown-ups): drag joints, then rebuild the rig ---

function drawJoints() {
  const c = $("joints-layer"), ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  if (!state.editing) return;
  const k = c.width / parseFloat($("canvas-box").style.width);
  ctx.setTransform(k, 0, 0, k, 0, 0);
  for (const ch of state.analysis.characters) {
    const by = Object.fromEntries(ch.joints.map((j) => [j.name, j]));
    ctx.lineWidth = 4;
    ctx.strokeStyle = "rgba(255,255,255,.9)";
    for (const j of ch.joints) {
      if (!by[j.parent]) continue;
      const [ax, ay] = toBox([by[j.parent].x, by[j.parent].y]), [bx, by_] = toBox([j.x, j.y]);
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by_);
      ctx.stroke();
    }
    for (const j of ch.joints) {
      const [x, y] = toBox([j.x, j.y]);
      ctx.fillStyle = j === state.dragging ? "#ff5fd2" : j.parent ? "#ffd86b" : "#7cf";
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#2a1650";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0);
}

function pointerToScan(e) {
  const r = $("joints-layer").getBoundingClientRect();
  return [(e.clientX - r.left) / state.scale + state.view[0], (e.clientY - r.top) / state.scale + state.view[1]];
}

$("joints-layer").addEventListener("pointerdown", (e) => {
  if (!state?.editing) return;
  const [x, y] = pointerToScan(e);
  let best = null, bestD = 18 / state.scale;
  for (const ch of state.analysis.characters)
    for (const j of ch.joints) {
      const d = Math.hypot(j.x - x, j.y - y);
      if (d < bestD) (best = j), (bestD = d);
    }
  state.dragging = best;
  if (best) $("joints-layer").setPointerCapture(e.pointerId);
  drawJoints();
});
$("joints-layer").addEventListener("pointermove", (e) => {
  if (!state?.dragging) return;
  const [x, y] = pointerToScan(e);
  state.dragging.x = Math.round(Math.max(0, Math.min(state.scan.width - 1, x)));
  state.dragging.y = Math.round(Math.max(0, Math.min(state.scan.height - 1, y)));
  drawJoints();
});
$("joints-layer").addEventListener("pointerup", () => {
  state.dragging = null;
  drawJoints();
});

async function toggleEdit() {
  const btn = $("btn-edit");
  if (!state.editing) {
    state.editing = true;
    btn.textContent = "✅ 调好了";
    $("joints-layer").classList.add("editing");
    say("拖动圆点，把关节放到身体里正确的位置");
    drawJoints();
    return;
  }
  btn.disabled = true;
  say("🪄 重新施魔法…");
  try {
    await loadRig(await api("/api/rig", { id: state.scan.id, characters: state.analysis.characters }));
    state.editing = false;
    btn.textContent = "🔧 调骨骼";
    $("joints-layer").classList.remove("editing");
    drawJoints();
    play(state.motion);
    say(state.analysis.narration_zh);
  } catch (e) {
    say(`😢 ${e.message}`, { error: true });
  } finally {
    btn.disabled = false;
  }
}

// --- Inputs: file picker, drag & drop, camera ---

$("btn-file").onclick = () => $("file").click();
$("file").onchange = (e) => start(e.target.files[0]);
$("btn-gif").onclick = saveGif;
$("btn-edit").onclick = toggleEdit;
$("btn-again").onclick = () => {
  stopAnimation();
  if (state) state.editing = false;
  $("joints-layer").classList.remove("editing");
  $("btn-edit").textContent = "🔧 调骨骼";
  drawJoints();
  $("controls").hidden = true;
  $("file").value = "";
  say("");
  show("home");
};

const frameEl = $("frame");
frameEl.addEventListener("dragover", (e) => {
  e.preventDefault();
  frameEl.classList.add("dragover");
});
frameEl.addEventListener("dragleave", () => frameEl.classList.remove("dragover"));
frameEl.addEventListener("drop", (e) => {
  e.preventDefault();
  frameEl.classList.remove("dragover");
  start(e.dataTransfer.files[0]);
});

let stream = null;
$("btn-camera").onclick = async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1920 }, height: { ideal: 1440 } } });
  } catch {
    say("😢 打不开摄像头，可以用「选照片」", { error: true });
    return;
  }
  $("video").srcObject = stream;
  show("camera");
};
const stopCamera = () => stream?.getTracks().forEach((t) => t.stop());
$("btn-cancel-camera").onclick = () => {
  stopCamera();
  show("home");
};
$("btn-snap").onclick = () => {
  const v = $("video"), c = document.createElement("canvas");
  c.width = v.videoWidth;
  c.height = v.videoHeight;
  c.getContext("2d").drawImage(v, 0, 0);
  stopCamera();
  c.toBlob((b) => start(new File([b], "camera.jpg", { type: "image/jpeg" })), "image/jpeg", 0.92);
};

window.addEventListener("resize", () => {
  if (!state?.scene) return;
  layout(state.view, false);
  sparkles.resize();
  drawJoints();
});
