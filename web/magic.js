// Sparkle effects on a 2D canvas laid over the picture. Coordinates are CSS pixels of that canvas.

const COLORS = ["#fff6a8", "#ffd86b", "#ff9fe0", "#c9b8ff", "#ffffff", "#9ff3ff"];
const pick = (a) => a[Math.floor(Math.random() * a.length)];

function star(ctx, x, y, r, color, alpha) {
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.shadowColor = color;
  ctx.shadowBlur = r * 3;
  ctx.beginPath();
  for (let i = 0; i < 8; i++) {
    const a = (i * Math.PI) / 4, rr = i % 2 ? r * 0.38 : r;
    ctx.lineTo(x + rr * Math.cos(a), y + rr * Math.sin(a));
  }
  ctx.fill();
}

export class Sparkles {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.parts = [];
    this.emitters = [];
    const loop = (now) => {
      this.tick(now);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  resize() {
    const dpr = window.devicePixelRatio || 1, r = this.canvas.getBoundingClientRect();
    this.canvas.width = r.width * dpr;
    this.canvas.height = r.height * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = r.width;
    this.h = r.height;
  }

  add(x, y, opts = {}) {
    const a = Math.random() * Math.PI * 2, sp = opts.speed ?? 0.3;
    this.parts.push({
      x, y, vx: Math.cos(a) * sp * Math.random(), vy: Math.sin(a) * sp * Math.random() - (opts.rise ?? 0.05),
      r: opts.r ?? 2 + Math.random() * 4, color: pick(COLORS), life: 0, ttl: opts.ttl ?? 900 + Math.random() * 700,
    });
  }

  // Gentle glitter drifting over the whole picture while we wait for the magic.
  glimmer(on) {
    this.emitters = this.emitters.filter((e) => e.kind !== "glimmer");
    if (on) this.emitters.push({ kind: "glimmer", rate: 0.04, emit: () => this.add(Math.random() * this.w, Math.random() * this.h, { speed: 0.1, ttl: 1600 }) });
  }

  // A comet of sparkles running around an outline (points in canvas pixels), resolves when done.
  trace(points, ms = 2200) {
    if (!points.length) return Promise.resolve();
    return new Promise((done) => {
      const start = performance.now();
      const e = {
        kind: "trace", rate: 1.2,
        emit: (now) => {
          const k = Math.max(0, now - start) / ms; // rAF time can be slightly earlier than start
          if (k >= 1) {
            this.emitters = this.emitters.filter((x) => x !== e);
            done();
            return;
          }
          const [x, y] = points[Math.floor(k * points.length)];
          this.add(x, y, { speed: 0.25, r: 2 + Math.random() * 5, ttl: 700 + Math.random() * 900 });
          // A few stay behind along the outline so it glows all the way round.
          const [qx, qy] = pick(points.slice(0, Math.max(1, Math.floor(k * points.length))));
          this.add(qx, qy, { speed: 0.05, r: 1.5 + Math.random() * 2, ttl: 600 });
        },
      };
      this.emitters.push(e);
    });
  }

  burst(x, y, n = 70) {
    for (let i = 0; i < n; i++) this.add(x, y, { speed: 2.5 + Math.random() * 3, r: 3 + Math.random() * 6, ttl: 900 + Math.random() * 800, rise: 0 });
  }

  tick(now) {
    const dt = Math.min(50, now - (this.last ?? now));
    this.last = now;
    for (const e of this.emitters) {
      e.acc = (e.acc ?? 0) + e.rate * dt;
      while (e.acc >= 1) {
        e.acc -= 1;
        e.emit(now);
      }
    }
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.w ?? 0, this.h ?? 0);
    ctx.globalCompositeOperation = "lighter";
    this.parts = this.parts.filter((p) => (p.life += dt) < p.ttl);
    for (const p of this.parts) {
      p.x += p.vx * dt * 0.06;
      p.y += p.vy * dt * 0.06;
      p.vx *= 0.985;
      p.vy = p.vy * 0.985 + 0.0008 * dt;
      const k = p.life / p.ttl;
      star(ctx, p.x, p.y, p.r * (1 - k * 0.6), p.color, Math.sin(Math.PI * Math.min(1, k * 1.2)));
    }
    ctx.globalAlpha = 1;
    ctx.shadowBlur = 0;
    ctx.globalCompositeOperation = "source-over";
  }
}

// Natural-sounding voices first (Edge/Windows "Natural", macOS Samantha, Chrome's Google voice).
const VOICE_PREFERENCE = [/natural/i, /samantha/i, /google us english/i, /aria|jenny/i];

export function speak(text) {
  if (!("speechSynthesis" in window) || !text) return;
  const u = new SpeechSynthesisUtterance(text);
  u.lang = "en-US";
  const english = speechSynthesis.getVoices().filter((v) => v.lang.startsWith("en"));
  const voice = VOICE_PREFERENCE.map((re) => english.find((v) => re.test(v.name))).find(Boolean) ?? english[0];
  if (voice) u.voice = voice;
  u.rate = 0.95;
  u.pitch = 1.1;
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
}

export function starrySky(el, n = 90) {
  for (let i = 0; i < n; i++) {
    const s = document.createElement("div");
    const size = Math.random() < 0.85 ? 1 + Math.random() * 2 : 3 + Math.random() * 2;
    s.className = "star";
    Object.assign(s.style, { left: `${Math.random() * 100}%`, top: `${Math.random() * 100}%`, width: `${size}px`, height: `${size}px` });
    s.style.setProperty("--d", `${2 + Math.random() * 4}s`);
    s.style.setProperty("--delay", `${-Math.random() * 6}s`);
    el.appendChild(s);
  }
}
