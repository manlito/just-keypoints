// The live demo: two images, their keypoints, the matches between them, and the time each step takes.
import { jkp, environment, weights, pixels, loadImage, sourceSize, ms, unsupportedMessage } from '../common.js';

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
  results: [null, null],     // { n, dim, xy, score, desc, w, h, prepMs, extractMs }
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
    mode,
    topK: +$('topk').value,
    th: +$('th').value,
    cap: +$('cap').value,
    res: +$('res').value,
    precision: document.querySelector('input[name="prec"]:checked').value,
    backend: document.querySelector('input[name="backend"]:checked').value,
    ratio: +$('ratio').value,
  };
}

function extractorKey(s) {
  return s.mode === 'topk' ? `${s.backend}|${s.precision}|k${s.topK}` : `${s.backend}|${s.precision}|t${s.th}|${s.cap}`;
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
      if (level >= LEVEL.match) await matchAll();
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
  const options = { weights: await weights(), backend: s.backend, precision: s.precision };
  if (s.mode === 'topk') options.topK = s.topK;
  else Object.assign(options, { scoreThreshold: s.th, maxKeypoints: s.cap });
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
    draw();
  }
}

let worker = null, nextId = 1;

function matchAll() {
  const [a, b] = state.results;
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
    const res = state.results[i], r = rects[i];
    return [r.x + (res.xy[2 * k] + 0.5) * r.w / res.w, r.y + (res.xy[2 * k + 1] + 0.5) * r.h / res.h];
  };
  if ($('show-kp').checked) {
    const radius = Math.max(1.2, Math.min(2.2, rects[0].w / 400));
    for (let i = 0; i < 2; i++) {
      const res = state.results[i];
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
  const [a, b] = state.results;
  const rows = [];
  const cell = (r, f) => (r ? f(r) : '–');
  rows.push(['Size', cell(a, (r) => `${r.w}x${r.h}`), cell(b, (r) => `${r.w}x${r.h}`)]);
  rows.push(['Prepare', cell(a, (r) => ms(r.prepMs)), cell(b, (r) => ms(r.prepMs))]);
  rows.push(['Extract', cell(a, (r) => ms(r.extractMs)), cell(b, (r) => ms(r.extractMs))]);
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
  const kv = [
    ['Backend', s.backend === 'webgpu' ? 'WebGPU' : 'CPU (WebAssembly, one thread)'],
    ['GPU', env?.webgpu ? env.adapter : 'no WebGPU adapter'],
    ['Precision', s.precision.toUpperCase()],
    ['Selection', s.mode === 'topk' ? `the best ${s.topK}` : `score above ${s.th.toFixed(2)}, at most ${s.cap}`],
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

  for (const r of document.querySelectorAll('input[name="mode"]')) {
    r.addEventListener('change', () => {
      const topk = settings().mode === 'topk';
      $('f-topk').hidden = !topk;
      $('f-th').hidden = topk;
      request('extract');
    });
  }
  for (const id of ['topk', 'th', 'cap', 'res']) $(id).addEventListener('change', () => request('extract'));
  for (const r of document.querySelectorAll('input[name="prec"]')) r.addEventListener('change', () => request('extract'));
  for (const r of document.querySelectorAll('input[name="backend"]')) {
    r.addEventListener('change', () => {
      const s = settings();
      if (s.backend === 'cpu' && (s.res === 0 || s.res > 640)) $('res').value = '640';
      if (s.backend === 'cpu') document.querySelector('input[name="prec"][value="fp32"]').checked = true;
      document.querySelector('input[name="prec"][value="fp16"]').disabled = s.backend === 'cpu' || !state.env.f16;
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
  busy('Loading the module…');
  await Promise.all([jkp.module(), weights(), loadPair('tum')]);
  request('extract');
}

main().catch((e) => {
  console.error(e);
  busy(null);
  showBanner(`The demo could not start: ${e.message ?? e}`, 'warn');
});
