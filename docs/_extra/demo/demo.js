// The live demo: two images, their keypoints, the matches between them, and the time each step takes.
import { jkp, environment, weights, RACO_WEIGHTS_URL, XFEAT_WEIGHTS_URL, pixels, loadImage, sourceSize, ms, unsupportedMessage } from '../common.js';

const $ = (id) => document.getElementById(id);
const SAMPLES = {
  tum: [['../samples/tum-1.jpg', 'TUM RGB-D, frame 1'], ['../samples/tum-2.jpg', 'TUM RGB-D, frame 2']],
  kitti: [['../samples/kitti-1.jpg', 'KITTI, frame 1'], ['../samples/kitti-2.jpg', 'KITTI, frame 2']],
};
// Longest side the page sends to the library. The WebGPU backend keeps full-resolution feature maps on the
// GPU, and the CPU backend in WebAssembly has at most 4 GB of memory (about 1.5 megapixels).
const MAX_SIDE = { webgpu: 2560, cpu: 1024 };
const LEVEL = { draw: 0, match: 1, extract: 2 };

const state = {
  env: null,
  images: [null, null],      // { bitmap, name }
  results: [null, null],     // { n, dim, xy, score, desc, rank, cov, w, h, prepMs, extractMs }
  shown: [null, null],       // the results drawn and matched: RaCo's best by ranker score, otherwise results
  firstMs: null,             // the first run after creating an extractor (compiles the shaders)
  matches: null,             // { pairs, sims, ms }
  extractor: null,
  key: '',
  fresh: false,
  warm: new Set(),           // sizes already run on this extractor
  error: null,
};

// For checks from the console or a test driver: the page's state, and `running` while a step runs.
window.jkpDemo = state;

// ---------- settings ----------

function settings() {
  const mode = document.querySelector('input[name="mode"]:checked').value;
  return {
    detector: document.querySelector('input[name="det"]:checked').value,
    mode,
    topK: +$('topk').value,
    rankKeep: +$('rank').value,
    th: +$('th').value,
    cap: +$('cap').value,
    res: +$('res').value,
    precision: document.querySelector('input[name="prec"]:checked').value,
    backend: document.querySelector('input[name="backend"]:checked').value,
    ratio: +$('ratio').value,
  };
}

function extractorKey(s) {
  if (s.detector === 'raco') return `raco|${s.backend}|k${s.topK}`;
  if (s.detector === 'xfeat') return `xfeat|${s.backend}|${s.precision}|k${s.topK}`;
  return s.mode === 'topk' ? `${s.backend}|${s.precision}|k${s.topK}` : `${s.backend}|${s.precision}|t${s.th}|${s.cap}`;
}

// RaCo: the `keep` keypoints with the highest ranker scores, best first (0 or more than detected: all).
function byRanker(r, keep) {
  if (!r.rank || keep <= 0 || keep >= r.n) return r;
  const order = Array.from({ length: r.n }, (_, i) => i).sort((a, b) => r.rank[b] - r.rank[a] || a - b).slice(0, keep);
  const pick = (src, width) => {
    const out = new Float32Array(order.length * width);
    order.forEach((k, i) => out.set(src.subarray(k * width, (k + 1) * width), i * width));
    return out;
  };
  return { ...r, n: order.length, xy: pick(r.xy, 2), score: pick(r.score, 1), desc: pick(r.desc, r.dim),
           rank: pick(r.rank, 1), cov: r.cov && pick(r.cov, 4), detected: r.n };
}

function selectAll() {
  const s = settings();
  state.shown = state.results.map((r) => r && (s.detector === 'raco' ? byRanker(r, s.rankKeep) : r));
}

function targetSize(image, s) {
  const { w, h } = sourceSize(image.bitmap);
  let side = Math.min(s.res > 0 ? s.res : Infinity, MAX_SIDE[s.backend]);
  const scale = Math.min(1, side / Math.max(w, h));
  return { w: Math.max(32, Math.round(w * scale)), h: Math.max(32, Math.round(h * scale)), capped: s.res === 0 && scale < 1 };
}

// ---------- running: one step at a time, the latest request wins ----------

let running = false, pending = -1;

function request(level) {
  pending = Math.max(pending, LEVEL[level]);
  if (!running) pump();
}

async function pump() {
  running = state.running = true;
  $('run').disabled = true;
  while (pending >= 0) {
    const level = pending;
    pending = -1;
    try {
      state.error = null;
      state.step = level;
      if (level >= LEVEL.extract) await extractAll();
      if (level >= LEVEL.match) {
        selectAll();
        await matchAll();
      }
    } catch (e) {
      console.error(e);
      state.error = e;
      showBanner(`Something went wrong: ${e.message ?? e}`, 'warn');
    }
    busy(null);
    draw();
    renderStats();
  }
  running = state.running = false;
  $('run').disabled = !state.images[0] || !state.images[1];
}

async function ensureExtractor(s) {
  const key = extractorKey(s);
  if (state.extractor && state.key === key) return;
  if (state.extractor) state.extractor.destroy();
  state.extractor = null;
  busy('Preparing the backend…');
  let options;
  if (s.detector === 'raco') {
    // RaCo keypoints, described by ALIKED for matching; RaCo runs in FP32 only.
    options = { weights: await weights(RACO_WEIGHTS_URL), descriptorWeights: await weights(), backend: s.backend,
                precision: 'fp32', topK: s.topK };
  } else if (s.detector === 'xfeat') {
    // XFeat keeps the best K above its own detection threshold (0.05).
    options = { weights: await weights(XFEAT_WEIGHTS_URL), backend: s.backend, precision: s.precision, topK: s.topK };
  } else {
    options = { weights: await weights(), backend: s.backend, precision: s.precision };
    if (s.mode === 'topk') options.topK = s.topK;
    else Object.assign(options, { scoreThreshold: s.th, maxKeypoints: s.cap });
  }
  state.extractor = await jkp.create(options);
  state.key = key;
  state.fresh = true;
}

async function extractAll() {
  if (!state.images[0] || !state.images[1]) return;
  const s = settings();
  await ensureExtractor(s);
  state.matches = null;
  state.results = [null, null];
  state.shown = [null, null];
  for (let i = 0; i < 2; i++) {
    const image = state.images[i];
    const { w, h } = targetSize(image, s);
    let t = performance.now();
    const data = pixels(image.bitmap, w, h);
    const prepMs = performance.now() - t;
    // Untimed runs: the first on a new extractor compiles the GPU shaders, and the first at a new size
    // allocates its buffers. The CPU backend has neither cost.
    if (state.fresh) {
      state.warm = new Set();
      state.firstMs = null;
    }
    if (s.backend === 'webgpu' && !state.warm.has(`${w}x${h}`)) {
      busy(state.fresh ? 'First run: compiling the GPU shaders…' : `First run at ${w}x${h}…`);
      t = performance.now();
      await state.extractor.extract(data);
      if (state.fresh) state.firstMs = performance.now() - t;
      await state.extractor.extract(data);
      state.warm.add(`${w}x${h}`);
    }
    state.fresh = false;
    busy(`Extracting ${'AB'[i]}…`);
    t = performance.now();
    const r = await state.extractor.extract(data);
    state.results[i] = { ...r, w, h, prepMs, extractMs: performance.now() - t };
    selectAll();
    draw();
  }
}

let worker = null, nextId = 1;

function matchAll() {
  const [a, b] = state.shown;
  if (!a || !b || !a.dim || !b.dim) { state.matches = null; return; }
  busy('Matching…');
  worker ??= new Worker(new URL('./matcher.js', import.meta.url));
  const id = nextId++;
  return new Promise((resolve, reject) => {
    worker.onmessage = (e) => { if (e.data.id === id) { state.matches = e.data; resolve(); } };
    worker.onerror = (e) => reject(new Error(e.message || 'the matcher failed'));
    worker.postMessage({ id, a: a.desc, b: b.desc, na: a.n, nb: b.n, dim: a.dim, ratio: settings().ratio });
  });
}

// ---------- drawing ----------

let rects = [];

function layout(width) {
  const [A, B] = state.images.map((im) => im && sourceSize(im.bitmap));
  if (!A || !B) return { rects: [], height: 0 };
  const gap = 8, aA = A.w / A.h, aB = B.w / B.h;
  if (Math.max(aA, aB) <= 1.6 && width >= 520) {           // side by side, same height
    const h = (width - gap) / (aA + aB);
    return { rects: [{ x: 0, y: 0, w: aA * h, h }, { x: aA * h + gap, y: 0, w: aB * h, h }], height: h };
  }
  const hA = width / aA, hB = width / aB;                     // one above the other
  return { rects: [{ x: 0, y: 0, w: width, h: hA }, { x: 0, y: hA + gap, w: width, h: hB }], height: hA + gap + hB };
}

function draw() {
  const canvas = $('view'), stage = $('stage');
  const width = stage.clientWidth, dpr = window.devicePixelRatio || 1;
  const L = layout(width);
  rects = L.rects;
  if (!L.height) return;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(L.height * dpr);
  canvas.style.height = `${L.height}px`;
  const g = canvas.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, width, L.height);
  for (let i = 0; i < 2; i++) {
    const r = rects[i];
    g.drawImage(state.images[i].bitmap, r.x, r.y, r.w, r.h);
  }
  // Keypoint k of image i on the canvas. xy is in the pixels sent to the library, (0, 0) = centre of the
  // top-left pixel.
  const at = (i, k) => {
    const res = state.shown[i], r = rects[i];
    return [r.x + (res.xy[2 * k] + 0.5) * r.w / res.w, r.y + (res.xy[2 * k + 1] + 0.5) * r.h / res.h];
  };
  // RaCo covariances as ellipses of `sigmas` standard deviations, in canvas pixels: Sigma = R diag(l1, l2) R^T.
  const sigmas = $('cov').disabled ? 0 : +$('cov').value;
  if (sigmas > 0) {
    g.lineWidth = 0.9;
    g.strokeStyle = 'rgba(64, 200, 255, 0.75)';
    for (let i = 0; i < 2; i++) {
      const res = state.shown[i], r = rects[i];
      if (!res?.cov) continue;
      const scale = sigmas * r.w / res.w;
      g.beginPath();
      for (let k = 0; k < res.n; k++) {
        const a = res.cov[4 * k], b = res.cov[4 * k + 1], d = res.cov[4 * k + 3];
        const mid = (a + d) / 2, rad = Math.hypot((a - d) / 2, b);
        const r1 = scale * Math.sqrt(Math.max(mid + rad, 0)), r2 = scale * Math.sqrt(Math.max(mid - rad, 0));
        const [x, y] = at(i, k), theta = 0.5 * Math.atan2(2 * b, a - d);
        g.moveTo(x + r1 * Math.cos(theta), y + r1 * Math.sin(theta));
        g.ellipse(x, y, r1, r2, theta, 0, 2 * Math.PI);
      }
      g.stroke();
    }
  }
  if ($('show-kp').checked) {
    const radius = Math.max(1.2, Math.min(2.2, rects[0].w / 400));
    for (let i = 0; i < 2; i++) {
      const res = state.shown[i];
      if (!res) continue;
      g.beginPath();
      for (let k = 0; k < res.n; k++) {
        const [x, y] = at(i, k);
        g.moveTo(x + radius, y);
        g.arc(x, y, radius, 0, 2 * Math.PI);
      }
      g.fillStyle = '#ffd23f';
      g.fill();
      g.lineWidth = 0.6;
      g.strokeStyle = 'rgba(0, 0, 0, 0.55)';
      g.stroke();
    }
  }
  const m = state.matches;
  if (m && $('show-m').checked && m.pairs.length) {
    let lo = 1;
    for (const s of m.sims) lo = Math.min(lo, s);
    const span = Math.max(1e-6, 1 - lo);
    const alpha = Math.max(0.3, Math.min(0.85, 300 / m.sims.length));   // fainter when there are many lines
    g.lineWidth = 1.1;
    for (let p = 0; p < m.sims.length; p++) {
      const [x0, y0] = at(0, m.pairs[2 * p]), [x1, y1] = at(1, m.pairs[2 * p + 1]);
      const t = (m.sims[p] - lo) / span;                      // 0 = weakest match shown, 1 = identical descriptors
      g.strokeStyle = `hsla(${Math.round(10 + 120 * t)}, 90%, 52%, ${alpha})`;
      g.beginPath();
      g.moveTo(x0, y0);
      g.lineTo(x1, y1);
      g.stroke();
    }
  }
}

// ---------- statistics ----------

function renderStats() {
  const s = settings();
  const [a, b] = state.shown;
  const raco = s.detector === 'raco', xfeat = s.detector === 'xfeat';
  const rows = [];
  const cell = (r, f) => (r ? f(r) : '–');
  rows.push(['Size', cell(a, (r) => `${r.w}x${r.h}`), cell(b, (r) => `${r.w}x${r.h}`)]);
  rows.push(['Prepare', cell(a, (r) => ms(r.prepMs)), cell(b, (r) => ms(r.prepMs))]);
  rows.push(['Extract', cell(a, (r) => ms(r.extractMs)), cell(b, (r) => ms(r.extractMs))]);
  if (raco && (a?.detected || b?.detected))
    rows.push(['Detected', cell(a, (r) => r.detected ?? r.n), cell(b, (r) => r.detected ?? r.n)]);
  rows.push(['Keypoints', cell(a, (r) => r.n), cell(b, (r) => r.n)]);
  const tbody = $('times').tBodies[0];
  tbody.replaceChildren(...rows.map(([k, x, y]) => tr([k, x, y])));
  const m = state.matches;
  tbody.append(tr(['Match', m ? ms(m.ms) : '–'], true));
  if (a && b) {
    const total = a.prepMs + a.extractMs + b.prepMs + b.extractMs + (m ? m.ms : 0);
    const row = tr(['Total', ms(total)], true);
    row.className = 'total';
    tbody.append(row);
  }
  if (state.firstMs != null) tbody.append(tr(['First run (compiles shaders)', ms(state.firstMs)], true));

  const env = state.env;
  const selection = raco
    ? `the best ${s.topK} by detection score` + (s.rankKeep > 0 ? `, then the best ${s.rankKeep} by ranker score` : '')
    : xfeat ? `the best ${s.topK} above 0.05`
    : s.mode === 'topk' ? `the best ${s.topK}` : `score above ${s.th.toFixed(2)}, at most ${s.cap}`;
  const kv = [
    ['Detector', raco ? 'RaCo, ALIKED descriptors' : xfeat ? 'XFeat' : 'ALIKED'],
    ['Backend', s.backend === 'webgpu' ? 'WebGPU' : 'CPU (WebAssembly, one thread)'],
    ['GPU', env?.webgpu ? env.adapter : 'no WebGPU adapter'],
    ['Precision', raco ? 'FP32' : s.precision.toUpperCase()],
    ['Selection', selection],
    ['Descriptors', a ? `${a.dim} floats, L2-normalized` : '–'],
    ['Matches', m ? `${m.sims.length}, ratio ${s.ratio >= 1 ? 'off' : s.ratio.toFixed(2)}` : '–'],
  ];
  $('device').replaceChildren(...kv.flatMap(([k, v]) => [el('dt', k), el('dd', v)]));

  const legend = [];
  if (a) legend.push(['A', `${a.n} keypoints`]);
  if (b) legend.push(['B', `${b.n} keypoints`]);
  if (m) legend.push(['Matches', `${m.sims.length}`]);
  if (a && b) legend.push(['Extract A + B', `${ms(a.extractMs + b.extractMs)} ms`]);
  $('legend').replaceChildren(...legend.map(([k, v]) => { const span = el('span', `${k} `); span.append(el('b', v)); return span; }));
}

function el(tag, text) { const e = document.createElement(tag); e.textContent = text; return e; }
function tr(cells, span = false) {
  const row = document.createElement('tr');
  cells.forEach((c, i) => {
    const td = el('td', String(c));
    if (span && i === 1) td.colSpan = 2;
    row.append(td);
  });
  return row;
}

function busy(text) {
  $('busy').hidden = !text;
  if (text) $('busy').textContent = text;
}

function showBanner(text, kind = '') {
  const b = $('banner');
  b.hidden = !text;
  b.className = `banner ${kind}`;
  b.textContent = text ?? '';
}

// ---------- images ----------

function setImage(i, bitmap, name) {
  state.images[i] = { bitmap, name };
  state.results[i] = null;
  state.shown[i] = null;
  state.matches = null;
  const slot = document.querySelector(`.slot[data-slot="${i}"]`);
  const { w, h } = sourceSize(bitmap);
  slot.querySelector('.name').textContent = `${'AB'[i]} · ${name} · ${w}x${h}`;
  slot.querySelector('.name').title = name;
  const thumb = slot.querySelector('.thumb'), g = thumb.getContext('2d');
  const cover = Math.max(thumb.width / w, thumb.height / h);   // fill the thumbnail, cropping the overflow
  g.clearRect(0, 0, thumb.width, thumb.height);
  g.drawImage(bitmap, (thumb.width - w * cover) / 2, (thumb.height - h * cover) / 2, w * cover, h * cover);
}

async function loadPair(name) {
  busy('Loading the images…');
  const bitmaps = await Promise.all(SAMPLES[name].map(([url]) => loadImage(url)));
  for (let i = 0; i < 2; i++) setImage(i, bitmaps[i], SAMPLES[name][i][1]);
}

async function fromFile(i, file) {
  if (!file || !file.type.startsWith('image/')) return;
  const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
  setImage(i, bitmap, file.name);
  markCustom();
  request('extract');
}

function markCustom() {
  const opt = $('pair').querySelector('option[value="custom"]');
  opt.hidden = false;
  $('pair').value = 'custom';
}

let fileSlot = 0;
let stream = null, cameraSlot = 0;

async function openCamera(i) {
  cameraSlot = i;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  } catch (e) {
    showBanner(`The camera is not available: ${e.message ?? e}`, 'warn');
    return;
  }
  const video = $('video');
  video.srcObject = stream;
  $('camera').hidden = false;
  await video.play();
  $('cam-take').focus();
}

function closeCamera() {
  $('camera').hidden = true;
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  $('video').srcObject = null;
}

// ---------- wiring ----------

function wire() {
  const out = (id, f) => { const update = () => { $(`${id}-v`).textContent = f(+$(id).value); }; $(id).addEventListener('input', update); update(); };
  out('topk', (v) => v);
  out('th', (v) => v.toFixed(2));
  out('cap', (v) => v);
  out('ratio', (v) => (v >= 1 ? 'off' : v.toFixed(2)));
  out('rank', (v) => (v > 0 ? v : 'all'));

  for (const r of document.querySelectorAll('input[name="det"]')) {
    r.addEventListener('change', () => {
      syncControls();
      request('extract');
    });
  }
  $('rank').addEventListener('change', () => request('match'));
  $('cov').addEventListener('change', () => request('draw'));

  for (const r of document.querySelectorAll('input[name="mode"]')) {
    r.addEventListener('change', () => {
      syncControls();
      request('extract');
    });
  }
  for (const id of ['topk', 'th', 'cap', 'res']) $(id).addEventListener('change', () => request('extract'));
  for (const r of document.querySelectorAll('input[name="prec"]')) r.addEventListener('change', () => request('extract'));
  for (const r of document.querySelectorAll('input[name="backend"]')) {
    r.addEventListener('change', () => {
      const s = settings();
      if (s.backend === 'cpu' && (s.res === 0 || s.res > 640)) $('res').value = '640';
      syncControls();
      request('extract');
    });
  }
  $('ratio').addEventListener('change', () => request('match'));
  $('show-kp').addEventListener('change', () => request('draw'));
  $('show-m').addEventListener('change', () => request('draw'));
  $('run').addEventListener('click', () => request('extract'));
  $('pair').addEventListener('change', async () => {
    if ($('pair').value === 'custom') return;
    await loadPair($('pair').value);
    request('extract');
  });

  for (const slot of document.querySelectorAll('.slot')) {
    const i = +slot.dataset.slot;
    slot.querySelector('[data-act="file"]').addEventListener('click', () => { fileSlot = i; $('file').click(); });
    slot.querySelector('[data-act="camera"]').addEventListener('click', () => openCamera(i));
  }
  $('file').addEventListener('change', () => { fromFile(fileSlot, $('file').files[0]); $('file').value = ''; });
  if (!navigator.mediaDevices?.getUserMedia) {
    for (const b of document.querySelectorAll('[data-act="camera"]')) b.disabled = true;
  }
  $('cam-cancel').addEventListener('click', closeCamera);
  $('cam-take').addEventListener('click', async () => {
    const bitmap = await createImageBitmap($('video'));
    closeCamera();
    setImage(cameraSlot, bitmap, `camera, ${new Date().toLocaleTimeString()}`);
    markCustom();
    request('extract');
  });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !$('camera').hidden) closeCamera(); });

  const stage = $('stage');
  stage.addEventListener('dragover', (e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; });
  stage.addEventListener('drop', (e) => {
    e.preventDefault();
    const box = $('view').getBoundingClientRect();
    const x = e.clientX - box.left, y = e.clientY - box.top;
    const i = rects[1] && x >= rects[1].x && y >= rects[1].y ? 1 : 0;
    fromFile(i, e.dataTransfer.files[0]);
  });
  new ResizeObserver(() => draw()).observe(stage);
}

// Controls that depend on the detector, backend and selection mode.
function syncControls() {
  const s = settings(), raco = s.detector === 'raco', xfeat = s.detector === 'xfeat';
  $('f-mode').hidden = raco || xfeat;
  $('f-topk').hidden = !raco && !xfeat && s.mode !== 'topk';
  $('f-th').hidden = raco || xfeat || s.mode === 'topk';
  $('f-rank').hidden = !raco;
  $('topk-l').textContent = raco ? 'Detected per image' : 'Keypoints per image';
  $('cov').disabled = !raco;
  $('det-hint').textContent = raco
    ? 'RaCo detects keypoints and adds a ranker score and a covariance to each; ALIKED describes them for ' +
      'matching. Two networks per image, FP32 only.'
    : xfeat
    ? 'XFeat detects and describes keypoints (64 floats) in one light network: about a third of ALIKED\'s time on a CPU, half on a GPU.'
    : 'ALIKED detects and describes keypoints in one network.';
  $('backend-hint').textContent = raco
    ? 'The CPU backend runs in WebAssembly on one thread: RaCo with descriptors takes about 12 s per 640x480 image.'
    : xfeat
    ? 'The CPU backend runs in WebAssembly on one thread: XFeat takes about 0.4 s per 640x480 image.'
    : 'The CPU backend runs in WebAssembly on one thread: about a second per 640x480 image.';
  const fp16 = document.querySelector('input[name="prec"][value="fp16"]');
  fp16.disabled = raco || s.backend === 'cpu' || !state.env?.f16 || !state.env?.webgpu;
  if (fp16.disabled) document.querySelector('input[name="prec"][value="fp32"]').checked = true;
}

async function main() {
  wire();
  const env = state.env = await environment();
  const unsupported = unsupportedMessage(env);
  if (unsupported) {
    showBanner(unsupported, 'warn');
    busy(null);
    await loadPair('tum');
    draw();
    renderStats();
    return;
  }
  if (!env.webgpu) {
    showBanner('WebGPU is not available in this browser, so the demo runs on the CPU backend: one thread in ' +
               'WebAssembly, about a second per image at 640 px. Chrome or Edge with WebGPU runs it about 100 times faster.', 'warn');
    document.querySelector('input[name="backend"][value="webgpu"]').disabled = true;
    document.querySelector('input[name="backend"][value="cpu"]').checked = true;
    $('res').value = '640';
  }
  if (!env.f16 || !env.webgpu) {
    document.querySelector('input[name="prec"][value="fp16"]').disabled = true;
    $('prec-hint').textContent = env.webgpu
      ? 'This GPU adapter has no shader-f16 feature, so FP16 is not available. FP32 matches the original network.'
      : 'FP16 needs WebGPU with the shader-f16 feature. FP32 matches the original network.';
  }
  syncControls();
  busy('Loading the module…');
  await Promise.all([jkp.module(), weights(), loadPair('tum')]);
  request('extract');
}

main().catch((e) => {
  console.error(e);
  busy(null);
  showBanner(`The demo could not start: ${e.message ?? e}`, 'warn');
});
