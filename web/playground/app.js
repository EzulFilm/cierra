// Cierra Playground: the program's WGSL shader and its output transform (display_encode exactly in WGSL, aces2 as a
// 97^3 LUT) run here on WebGPU; the local
// server (tools/playground_server.py) only compiles the program and turns a dropped frame into ACEScct.
"use strict";

const DEFAULT = `cierra 0.1
name "warm, a little richer"
input linear_rec709
output aces2(display: srgb)

node balance = white_balance(temperature: 0.1)
node tone = contrast(amount: 1.08, pivot: 0.38)
node colour = saturation(amount: 1.1, protect_skin: 0.5)
node edges = vignette(amount: 0.15, softness: 0.7)

graph balance -> tone -> colour -> edges
`;

const lutShader = (LUT_N) => `
@group(0) @binding(0) var<storage, read> src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;
@group(0) @binding(2) var<uniform> size : vec2<u32>;
@group(0) @binding(3) var<storage, read> lut : array<f32>;
const N : u32 = ${LUT_N}u;
fn at(r : u32, g : u32, b : u32) -> vec3<f32> {
  let i = 3u * ((b * N + g) * N + r);
  return vec3<f32>(lut[i], lut[i + 1u], lut[i + 2u]);
}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let p = 3u * (gid.y * size.x + gid.x);
  let c = clamp(vec3<f32>(src[p], src[p + 1u], src[p + 2u]), vec3<f32>(0.0), vec3<f32>(1.0)) * f32(N - 1u);
  let i0 = min(vec3<u32>(floor(c)), vec3<u32>(N - 2u));
  let f = c - vec3<f32>(i0);
  let r = i0.x; let g = i0.y; let b = i0.z;
  let c000 = at(r, g, b); let c111 = at(r + 1u, g + 1u, b + 1u);
  var o : vec3<f32>;
  // tetrahedral interpolation (the same rule as the .cube readers cierra's LUT parity is stated for)
  if (f.x >= f.y) {
    if (f.y >= f.z) {        // r >= g >= b
      o = (1.0 - f.x) * c000 + (f.x - f.y) * at(r + 1u, g, b) + (f.y - f.z) * at(r + 1u, g + 1u, b) + f.z * c111;
    } else if (f.x >= f.z) { // r >= b > g
      o = (1.0 - f.x) * c000 + (f.x - f.z) * at(r + 1u, g, b) + (f.z - f.y) * at(r + 1u, g, b + 1u) + f.y * c111;
    } else {                 // b > r >= g
      o = (1.0 - f.z) * c000 + (f.z - f.x) * at(r, g, b + 1u) + (f.x - f.y) * at(r + 1u, g, b + 1u) + f.y * c111;
    }
  } else {
    if (f.z >= f.y) {        // b >= g > r
      o = (1.0 - f.z) * c000 + (f.z - f.y) * at(r, g, b + 1u) + (f.y - f.x) * at(r, g + 1u, b + 1u) + f.x * c111;
    } else if (f.z >= f.x) { // g > b >= r
      o = (1.0 - f.y) * c000 + (f.y - f.z) * at(r, g + 1u, b) + (f.z - f.x) * at(r, g + 1u, b + 1u) + f.x * c111;
    } else {                 // g > r > b
      o = (1.0 - f.y) * c000 + (f.y - f.x) * at(r, g + 1u, b) + (f.x - f.z) * at(r + 1u, g + 1u, b) + f.z * c111;
    }
  }
  dst[p] = o.x; dst[p + 1u] = o.y; dst[p + 2u] = o.z;
}`;

const $ = (id) => document.getElementById(id);
const S = { device: null, frame: null, luts: {}, pipelines: {}, compiled: null, last: null, busy: false, again: false };

function status(text, kind) { const s = $("status"); s.textContent = text; s.className = kind || ""; }

async function gpu() {
  if (S.device) return S.device;
  if (!navigator.gpu) throw new Error("no WebGPU in this browser");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("WebGPU found no adapter");
  S.device = await adapter.requestDevice({ requiredLimits: {
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxBufferSize: adapter.limits.maxBufferSize } });
  return S.device;
}

function pipeline(dev, code) {
  if (!S.pipelines[code]) {
    const module = dev.createShaderModule({ code });
    S.pipelines[code] = dev.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  }
  return S.pipelines[code];
}

// one compute pass: src (Float32Array, r,g,b interleaved) -> dst of the same size; extra = a 4th storage buffer
async function runPass(code, src, w, h, extra) {
  const dev = await gpu();
  const pipe = pipeline(dev, code);
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const a = dev.createBuffer({ size: src.byteLength, usage });
  dev.queue.writeBuffer(a, 0, src);
  const b = dev.createBuffer({ size: src.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const u = dev.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  dev.queue.writeBuffer(u, 0, new Uint32Array([w, h]));
  const entries = [{ binding: 0, resource: { buffer: a } }, { binding: 1, resource: { buffer: b } },
                   { binding: 2, resource: { buffer: u } }];
  let e = null;
  if (extra) {
    e = dev.createBuffer({ size: extra.byteLength, usage });
    dev.queue.writeBuffer(e, 0, extra);
    entries.push({ binding: 3, resource: { buffer: e } });
  }
  const group = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
  const read = dev.createBuffer({ size: src.byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = dev.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipe); pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
  pass.end();
  enc.copyBufferToBuffer(b, 0, read, 0, src.byteLength);
  dev.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(read.getMappedRange().slice(0));
  read.unmap();
  for (const x of [a, b, u, e, read]) if (x) x.destroy();
  return out;
}

// the output transform: {wgsl} for an exact shader, or {wgsl: the LUT shader, table} for a sampled one
// Two backends, one page. "server": tools/playground_server.py answers /api/*. "static": no server (a folder built by
// tools/build_playground_static.py): Cierra's own compiler (cierra.compile.web) runs in the browser in Pyodide, the
// built-in frames and the ACES 2.0 output LUTs are precomputed files (they need OpenColorIO, which has no WebAssembly
// build), and dropped PNG / JPEG frames are decoded here.
const B = { mode: null, py: null, manifest: null, frames: null };

function loadScript(url) {
  return new Promise((ok, bad) => { const s = document.createElement("script"); s.src = url; s.onload = ok;
    s.onerror = () => bad(new Error("could not load " + url)); document.head.appendChild(s); });
}

async function backend() {
  if (B.mode) return B;
  try {
    const r = await fetch("/api/frames");
    if (r.ok && (r.headers.get("content-type") || "").includes("json")) { B.mode = "server"; B.frames = await r.json(); return B; }
  } catch (_) { /* no server: static */ }
  B.mode = "static";
  status("loading the compiler in the browser (Pyodide, numpy, Cierra): a few seconds, once");
  B.manifest = await (await fetch("static/manifest.json")).json();
  await loadScript(B.manifest.pyodide + "pyodide.js");
  B.py = await loadPyodide({ indexURL: B.manifest.pyodide });
  await B.py.loadPackage("numpy");
  B.py.unpackArchive(await (await fetch("static/cierra.zip")).arrayBuffer(), "zip", { extractDir: "/home/pyodide/pkg" });
  B.py.runPython("import sys, json; sys.path.insert(0, '/home/pyodide/pkg'); from cierra.compile import web");
  B.frames = B.manifest.frames;
  return B;
}

function py(expr, vars) {
  for (const [k, v] of Object.entries(vars || {})) B.py.globals.set(k, v);
  return B.py.runPython(expr);
}

async function outputStage(output) {
  if (!S.luts[output]) {
    await backend();
    const q = encodeURIComponent(output);
    let o;
    if (B.mode === "server") {
      o = await (await fetch("/api/output?output=" + q)).json();
      if (o.error) throw new Error(o.error);
    } else {
      const code = py("web.output_shader(OUT)", { OUT: output });
      o = code ? { kind: "wgsl", wgsl: code } : { kind: "lut", ...(B.manifest.luts[output] || {}) };
      if (o.kind === "lut" && !o.file) throw new Error(`output ${output}: no precomputed LUT in this build (outputs: ${Object.keys(B.manifest.luts).join(", ")}, display_encode, acescct)`);
    }
    if (o.kind === "wgsl") {
      S.luts[output] = { wgsl: o.wgsl, table: null, kind: "exact" };
    } else {
      const r = await fetch(B.mode === "server" ? "/api/lut?output=" + q : "static/luts/" + o.file);
      if (!r.ok) throw new Error("output LUT: " + r.status);
      S.luts[output] = { wgsl: lutShader(o.n), table: new Float32Array(await r.arrayBuffer()), kind: `${o.n}^3 LUT` };
    }
  }
  return S.luts[output];
}

async function compile(src) {
  await backend();
  if (B.mode === "static") return JSON.parse(py("json.dumps(web.compile_program(SRC, 0))", { SRC: src }));
  const r = await fetch("/api/compile", { method: "POST", body: JSON.stringify({ program: src, frame: 0 }) });
  return r.json();
}

async function staticFrame(body) {
  if (body.builtin) {
    const f = B.frames.find((x) => x.name === body.builtin);
    if (!f) throw new Error("no built-in frame " + body.builtin);
    const c = await compile($("src").value);
    if (c.ok && c.photo_graph && f.input !== "linear_rec709") throw new Error("photo_graph needs OpenColorIO: run the local server for this program");
    const r = await fetch("static/frames/" + f.file);
    return r.arrayBuffer();
  }
  const bmp = await createImageBitmap(body.file);
  const cv = new OffscreenCanvas(bmp.width, bmp.height);
  const cx = cv.getContext("2d", { colorSpace: "srgb" });
  cx.drawImage(bmp, 0, 0);
  const rgba = cx.getImageData(0, 0, bmp.width, bmp.height).data;
  const out = py("web.decode_rgba8(PIX.to_py(), W, H, SRC)", { PIX: rgba, W: bmp.width, H: bmp.height, SRC: $("src").value });
  const bytes = out.toJs(); out.destroy();
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

async function loadFrame(body) {
  await backend();
  let buf;
  if (B.mode === "static") {
    buf = await staticFrame(body);
  } else {
    const r = await fetch("/api/frame", { method: "POST", body: JSON.stringify({ program: $("src").value, ...body }) });
    if (!r.ok) throw new Error((await r.json()).error);
    buf = await r.arrayBuffer();
  }
  const n = new DataView(buf).getUint32(0, true);
  const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, n)));
  S.frame = { ...head, code: new Float32Array(buf.slice(4 + n)), source: body.builtin || body.name };
  S.ungraded = null;
  return S.frame;
}

function paint(canvas, disp, w, h) {
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  for (let i = 0, j = 0; i < w * h; i++, j += 3) {
    img.data[4 * i] = Math.round(Math.min(1, Math.max(0, disp[j])) * 255);
    img.data[4 * i + 1] = Math.round(Math.min(1, Math.max(0, disp[j + 1])) * 255);
    img.data[4 * i + 2] = Math.round(Math.min(1, Math.max(0, disp[j + 2])) * 255);
    img.data[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

// scopes on the display code values (Rec.709 luma and Cb / Cr)
function scopes(disp, w, h) {
  const wave = $("wave"), vec = $("vec"), hist = $("hist");
  const W = wave.width, H = wave.height;
  const acc = new Float32Array(W * H), vacc = new Float32Array(W * H);
  const bins = [new Float32Array(256), new Float32Array(256), new Float32Array(256)];
  const step = Math.max(1, Math.floor((w * h) / 200000));
  for (let i = 0; i < w * h; i += step) {
    const r = disp[3 * i], g = disp[3 * i + 1], b = disp[3 * i + 2];
    const y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    const x = Math.floor(((i % w) / w) * W);
    const yy = H - 1 - Math.round(Math.min(1, Math.max(0, y)) * (H - 1));
    acc[yy * W + x] += 1;
    const cb = (b - y) / 1.8556, cr = (r - y) / 1.5748;
    const vx = Math.round(W / 2 + cb * H * 1.6), vy = Math.round(H / 2 - cr * H * 1.6);
    if (vx >= 0 && vx < W && vy >= 0 && vy < H) vacc[vy * W + vx] += 1;
    for (let c = 0; c < 3; c++) bins[c][Math.round(Math.min(1, Math.max(0, disp[3 * i + c])) * 255)] += 1;
  }
  const glow = (cv, data, tint) => {
    const ctx = cv.getContext("2d"), img = ctx.createImageData(cv.width, cv.height);
    let m = 0; for (const v of data) m = Math.max(m, v);
    const k = m > 0 ? 1 / Math.log1p(m) : 0;
    for (let i = 0; i < data.length; i++) {
      const v = Math.log1p(data[i]) * k;
      img.data[4 * i] = tint[0] * v; img.data[4 * i + 1] = tint[1] * v; img.data[4 * i + 2] = tint[2] * v; img.data[4 * i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return ctx;
  };
  const wc = glow(wave, acc, [120, 255, 140]);
  wc.strokeStyle = "rgba(255,255,255,0.15)";
  for (const t of [0, 0.25, 0.5, 0.75, 1]) { const y = H - 1 - t * (H - 1); wc.beginPath(); wc.moveTo(0, y); wc.lineTo(W, y); wc.stroke(); }
  const vc = glow(vec, vacc, [255, 255, 255]);
  vc.strokeStyle = "rgba(255,255,255,0.2)"; vc.beginPath(); vc.arc(W / 2, H / 2, H * 0.45, 0, 2 * Math.PI); vc.stroke();
  // the skin line (about 123 degrees on a Rec.709 vectorscope)
  const a = (123 * Math.PI) / 180; vc.beginPath(); vc.moveTo(W / 2, H / 2);
  vc.lineTo(W / 2 + Math.cos(a) * H * 0.45, H / 2 - Math.sin(a) * H * 0.45); vc.stroke();
  const hc = hist.getContext("2d"); hc.fillStyle = "#0b0d10"; hc.fillRect(0, 0, hist.width, hist.height);
  hc.globalCompositeOperation = "lighter";
  let m = 0; for (const bb of bins) for (let i = 1; i < 255; i++) m = Math.max(m, bb[i]);
  ["#ff5050", "#50ff70", "#5080ff"].forEach((col, c) => {
    hc.fillStyle = col + "aa";
    for (let i = 0; i < 256; i++) {
      const v = Math.min(1, bins[c][i] / (m || 1)); const x = (i / 256) * hist.width;
      hc.fillRect(x, hist.height * (1 - v), hist.width / 256 + 0.5, hist.height * v);
    }
  });
  hc.globalCompositeOperation = "source-over";
}

async function grade() {
  if (S.busy) { S.again = true; return; }
  S.busy = true;
  S.last = null;  // a failed grade must not leave the previous program's pixels as the result
  try {
    do {
      S.again = false;
      const src = $("src").value;
      const t0 = performance.now();
      const c = await compile(src);
      if (!c.ok) { status(c.errors.join("\n"), "bad"); S.compiled = null; continue; }
      S.compiled = c;
      if (!S.frame) await loadFrame({ builtin: $("builtin").value || "chart" });
      const f = S.frame, os = await outputStage(c.output);
      const code = await runPass(c.wgsl, f.code, f.width, f.height);
      const disp = await runPass(os.wgsl, code, f.width, f.height, os.table);
      if (!S.ungraded || S.ungraded.output !== c.output) {
        S.ungraded = { output: c.output, disp: await runPass(os.wgsl, f.code, f.width, f.height, os.table) };
      }
      paint($("before"), S.ungraded.disp, f.width, f.height);
      paint($("after"), disp, f.width, f.height);
      scopes(disp, f.width, f.height);
      S.last = { code, disp, width: f.width, height: f.height, output: c.output, frame: f.source };
      const ms = Math.round(performance.now() - t0);
      status(`ok (${B.mode}) · ${f.width}x${f.height} · ${f.source} (${f.input}) · output ${c.output} (${os.kind}) · ${ms} ms`
             + (c.warnings.length ? "\n" + c.warnings.join("\n") : ""), "ok");
    } while (S.again);
  } catch (e) {
    status(String(e.message || e), "bad");
  } finally {
    S.busy = false;
  }
}

function b64(bytes) {
  let s = ""; const u = new Uint8Array(bytes);
  for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000));
  return btoa(s);
}
const toHash = (src) => b64(new TextEncoder().encode(src)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
const fromHash = (h) => new TextDecoder().decode(Uint8Array.from(atob(h.replace(/-/g, "+").replace(/_/g, "/")), (c) => c.charCodeAt(0)));

async function init() {
  if (!navigator.gpu) $("nogpu").style.display = "block";
  const m = location.hash.match(/p=([A-Za-z0-9_-]+)/);
  $("src").value = m ? fromHash(m[1]) : DEFAULT;
  const frames = (await backend()).frames;
  for (const f of frames) $("builtin").add(new Option(`${f.name} (${f.input})`, f.name));
  $("builtin").value = "chart";
  $("builtin").onchange = async () => { await loadFrame({ builtin: $("builtin").value }); grade(); };
  $("run").onclick = grade;
  let timer = null;
  $("src").addEventListener("input", () => { if ($("live").checked) { clearTimeout(timer); timer = setTimeout(grade, 350); } });
  $("src").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); grade(); } });
  $("share").onclick = async () => {
    const url = location.origin + location.pathname + "#p=" + toHash($("src").value);
    history.replaceState(null, "", url);
    try { await navigator.clipboard.writeText(url); status("share link copied", "ok"); } catch (_) { status(url, "ok"); }
  };
  const take = async (file) => {
    try {
      if (B.mode === "static") await loadFrame({ name: file.name, file });
      else await loadFrame({ name: file.name, data: b64(await file.arrayBuffer()) });
      grade();
    } catch (e) { status(String(e.message || e), "bad"); }
  };
  const drop = $("drop");
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); if (e.dataTransfer.files[0]) take(e.dataTransfer.files[0]); });
  $("file").onchange = (e) => { if (e.target.files[0]) take(e.target.files[0]); };
  await grade();
}

// for the parity test (tools/playground_parity.py): grade a program on a built-in frame, hand back both buffers
window.cierraPlayground = {
  async run(program, builtin) {
    $("src").value = program;
    try { await loadFrame({ builtin }); } catch (e) { return { ok: false, status: String(e.message || e) }; }
    await grade();
    if (!S.compiled || !S.last) return { ok: false, status: $("status").textContent };
    return { ok: true, width: S.last.width, height: S.last.height, output: S.last.output,
             code: b64(S.last.code.buffer), disp: b64(S.last.disp.buffer), status: $("status").textContent };
  },
};

window.cierraPlaygroundReady = init().then(() => true, (e) => { status(String(e.message || e), "bad"); return false; });
