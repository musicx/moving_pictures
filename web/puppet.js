// Skeletal puppet: poses a rig from the server with a looping motion and draws it with WebGL.
// All geometry is in scan pixel coordinates; the renderer maps the scan onto its canvas.

const TAU = Math.PI * 2;

// 2D affine [a, b, c, d, e, f]: x' = a x + c y + e, y' = b x + d y + f
const IDENTITY = [1, 0, 0, 1, 0, 0];
const mul = (m, n) => [
  m[0] * n[0] + m[2] * n[1], m[1] * n[0] + m[3] * n[1],
  m[0] * n[2] + m[2] * n[3], m[1] * n[2] + m[3] * n[3],
  m[0] * n[4] + m[2] * n[5] + m[4], m[1] * n[4] + m[3] * n[5] + m[5],
];
const rotateAbout = (rad, x, y) => {
  const c = Math.cos(rad), s = Math.sin(rad);
  return [c, s, -s, c, x - c * x + s * y, y - s * x - c * y];
};
const scaleAbout = (sx, sy, x, y) => [sx, 0, 0, sy, x - sx * x, y - sy * y];

export function prepareRig(rig) {
  const index = new Map(rig.bones.map((b, i) => [b.joint, i]));
  const parentBone = rig.bones.map((b) => (b.parent === null ? -1 : index.get(b.parent)));
  const rest = Float32Array.from(rig.vertices);
  const uv = new Float32Array(rest.length);
  for (let i = 0; i < rest.length; i += 2) {
    uv[i] = (rest[i] - rig.x) / rig.w;
    uv[i + 1] = (rest[i + 1] - rig.y) / rig.h;
  }
  return { ...rig, index, parentBone, rest, uv, pos: new Float32Array(rest.length) };
}

// u in [0, 1): position in the loop. Every term is periodic in u, so u = 0 and u = 1 match exactly.
// strength scales every amplitude (0 = rest pose), used to ease the drawing into motion.
export function pose(rig, track, u, strength = 1) {
  const angle = new Float32Array(rig.bones.length);
  for (const b of track?.bones ?? []) {
    const k = rig.index.get(b.joint);
    if (k > 0) angle[k] = strength * (b.amp_deg * Math.PI / 180) * Math.sin(TAU * (b.cycles * u + b.phase));
  }
  const mats = [];
  rig.bones.forEach((b, k) => {
    const p = rig.parentBone[k];
    mats[k] = p < 0 ? IDENTITY : mul(mats[p], rotateAbout(angle[k], b.px, b.py));
  });

  let g = IDENTITY;
  if (track && strength > 0) {
    const c = track.cycles, ax = rig.x + rig.w / 2, ay = rig.y + rig.h;
    const wave = Math.sin(TAU * c * u);
    const air = Math.abs(Math.sin(Math.PI * c * u));
    const hop = strength * track.hop, breathe = strength * track.breathe;
    const land = hop > 0 ? 0.6 * hop * (1 - air) ** 6 : 0;
    const sy = (1 - land) * (1 + breathe * wave);
    const sx = (1 + land / 2) * (1 - breathe * wave / 2);
    const dy = -rig.h * strength * (track.hop * air + track.float * wave);
    g = mul([1, 0, 0, 1, 0, dy], mul(rotateAbout(strength * track.sway_deg * Math.PI / 180 * wave, ax, ay), scaleAbout(sx, sy, ax, ay)));
  }

  const { rest, pos, influences } = rig;
  const n = rest.length / 2, k = influences.length / n / 2;
  for (let i = 0; i < n; i++) {
    const x = rest[2 * i], y = rest[2 * i + 1];
    let ox = 0, oy = 0;
    for (let j = 0; j < k; j++) {
      const bone = influences[(i * k + j) * 2], w = influences[(i * k + j) * 2 + 1];
      const m = mats[bone];
      ox += w * (m[0] * x + m[2] * y + m[4]);
      oy += w * (m[1] * x + m[3] * y + m[5]);
    }
    pos[2 * i] = g[0] * ox + g[2] * oy + g[4];
    pos[2 * i + 1] = g[1] * ox + g[3] * oy + g[5];
  }
  return pos;
}

const VS = `
attribute vec2 a_pos; attribute vec2 a_uv; uniform vec4 u_view; varying vec2 v_uv;
void main() { vec2 p = (a_pos - u_view.xy) / u_view.zw * 2.0 - 1.0; gl_Position = vec4(p.x, -p.y, 0.0, 1.0); v_uv = a_uv; }`;
const FS = `
precision mediump float; uniform sampler2D u_tex; varying vec2 v_uv;
void main() { gl_FragColor = texture2D(u_tex, v_uv); }`;

export function loadImage(src) {
  return new Promise((ok, fail) => {
    const img = new Image();
    img.onload = () => ok(img);
    img.onerror = fail;
    img.src = src;
  });
}

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = (this.gl = canvas.getContext("webgl", { premultipliedAlpha: true, preserveDrawingBuffer: true, antialias: true }));
    const prog = gl.createProgram();
    for (const [type, src] of [[gl.VERTEX_SHADER, VS], [gl.FRAGMENT_SHADER, FS]]) {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      gl.attachShader(prog, s);
    }
    gl.linkProgram(prog);
    gl.useProgram(prog);
    this.loc = {
      pos: gl.getAttribLocation(prog, "a_pos"), uv: gl.getAttribLocation(prog, "a_uv"),
      view: gl.getUniformLocation(prog, "u_view"),
    };
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
  }

  texture(img) {
    const gl = this.gl, t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    return t;
  }

  // scene: { width, height, view: [x0, y0, x1, y1], background: Image, characters: [{ rig (prepared), image: Image }] }
  async setScene(scene) {
    const gl = this.gl;
    this.scene = scene;
    const W = scene.width, H = scene.height;
    this.bg = {
      tex: this.texture(scene.background),
      pos: Float32Array.from([0, 0, W, 0, 0, H, W, H]),
      uv: Float32Array.from([0, 0, 1, 0, 0, 1, 1, 1]),
      idx: Uint16Array.from([0, 1, 2, 2, 1, 3]),
    };
    this.chars = scene.characters.map((c) => ({
      rig: c.rig, tex: this.texture(c.image), uv: c.rig.uv, idx: Uint16Array.from(c.rig.triangles),
    }));
    this.buf = { pos: gl.createBuffer(), uv: gl.createBuffer(), idx: gl.createBuffer() };
  }

  drawMesh(tex, pos, uv, idx) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.pos);
    gl.bufferData(gl.ARRAY_BUFFER, pos, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(this.loc.pos);
    gl.vertexAttribPointer(this.loc.pos, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.uv);
    gl.bufferData(gl.ARRAY_BUFFER, uv, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(this.loc.uv);
    gl.vertexAttribPointer(this.loc.uv, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.buf.idx);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.DYNAMIC_DRAW);
    gl.drawElements(gl.TRIANGLES, idx.length, gl.UNSIGNED_SHORT, 0);
  }

  // motion: one of the analysis motions (or null for the rest pose); u in [0, 1)
  draw(motion, u, strength = 1) {
    const gl = this.gl, s = this.scene;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(1, 1, 1, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    const [x0, y0, x1, y1] = s.view;
    gl.uniform4f(this.loc.view, x0, y0, x1 - x0, y1 - y0);
    this.drawMesh(this.bg.tex, this.bg.pos, this.bg.uv, this.bg.idx);
    // Main character is first in the list; draw it last so it is on top.
    for (const c of [...this.chars].reverse()) {
      const track = motion?.tracks.find((t) => t.character_id === c.rig.id) ?? null;
      this.drawMesh(c.tex, pose(c.rig, track, u, strength), c.uv, c.idx);
    }
  }
}

export async function buildScene(scan, view, rigResult) {
  const [background, ...images] = await Promise.all([
    loadImage(rigResult.background), ...rigResult.characters.map((c) => loadImage(c.texture)),
  ]);
  return {
    width: scan.width, height: scan.height, view, background,
    characters: rigResult.characters.map((c, i) => ({ rig: prepareRig(c), image: images[i] })),
  };
}

const GIF_MS = 60; // per-frame delay; a multiple of 10 ms because GIF stores centiseconds

export async function exportGif(scene, motion, longSide = 480) {
  const { GIFEncoder, quantize, applyPalette } = await import("https://cdn.jsdelivr.net/npm/gifenc@1.0.3/dist/gifenc.esm.js");
  const [x0, y0, x1, y1] = scene.view;
  const k = longSide / Math.max(x1 - x0, y1 - y0);
  const canvas = document.createElement("canvas");
  canvas.width = Math.round((x1 - x0) * k);
  canvas.height = Math.round((y1 - y0) * k);
  const r = new Renderer(canvas);
  await r.setScene(scene);
  const n = Math.max(2, Math.round((motion.duration_s * 1000) / GIF_MS));
  const { width: w, height: h } = canvas, gl = r.gl;
  const frames = [];
  for (let i = 0; i < n; i++) {
    r.draw(motion, i / n);
    const px = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
    const flipped = new Uint8Array(px.length); // WebGL rows are bottom-up
    for (let y = 0; y < h; y++) flipped.set(px.subarray((h - 1 - y) * w * 4, (h - y) * w * 4), y * w * 4);
    frames.push(flipped);
  }
  // One palette for the whole loop keeps colours from flickering between frames.
  const sample = new Uint8Array(frames.length > 3 ? w * h * 4 * 3 : w * h * 4 * frames.length);
  [0, Math.floor(n / 3), Math.floor((2 * n) / 3)].slice(0, frames.length).forEach((f, j) => sample.set(frames[f], j * w * h * 4));
  const palette = quantize(sample, 256);
  const gif = GIFEncoder();
  for (const f of frames) gif.writeFrame(applyPalette(f, palette), w, h, { palette, delay: GIF_MS });
  gif.finish();
  return new Blob([gif.bytes()], { type: "image/gif" });
}
