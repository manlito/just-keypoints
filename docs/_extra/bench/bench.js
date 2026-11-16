// The benchmark page: one image at four sizes, the 1000 best keypoints, median of warm runs. The sizes are the
// ones of the native benchmark tables; each is a centre crop of the test image to the size's aspect ratio,
// scaled to the size.
import { jkp, environment, weights, pixels, loadImage, median, ms, unsupportedMessage } from '../common.js';

const $ = (id) => document.getElementById(id);
const SIZES = [[640, 480], [1241, 376], [1024, 768], [1920, 1080]];
const TOP_K = 1000;
let env = null, stopping = false, last = null;

function el(tag, text, cls) { const e = document.createElement(tag); e.textContent = text; if (cls) e.className = cls; return e; }

function browserName() {
  const brands = navigator.userAgentData?.brands?.filter((b) => !/Not.?A.?Brand/i.test(b.brand));
  if (brands?.length) return brands.map((b) => `${b.brand} ${b.version}`).join(', ');
  return navigator.userAgent;
}

function renderEnv() {
  const kv = [
    ['Browser', browserName()],
    ['Platform', navigator.userAgentData?.platform || navigator.platform || '–'],
    ['GPU', env.webgpu ? env.adapter : 'no WebGPU adapter'],
    ['shader-f16', env.webgpu ? (env.f16 ? 'yes' : 'no') : '–'],
    ['Logical CPUs', String(navigator.hardwareConcurrency ?? '–')],
  ];
  $('env').replaceChildren(...kv.flatMap(([k, v]) => [el('dt', k), el('dd', v)]));
}

function columns() {
  const backend = document.querySelector('input[name="backend"]:checked').value;
  const cols = [];
  if ($('fp32').checked) cols.push({ backend, precision: 'fp32' });
  if ($('fp16').checked && backend === 'webgpu' && env.f16) cols.push({ backend, precision: 'fp16' });
  return cols;
}

const label = (c) => `${c.backend === 'webgpu' ? 'WebGPU' : 'CPU'} ${c.precision.toUpperCase()}`;

function render(cols, table) {
  const thead = $('results').tHead, tbody = $('results').tBodies[0];
  const head = document.createElement('tr');
  head.append(el('th', 'Size'), ...cols.map((c) => el('th', label(c))), el('th', 'Keypoints'));
  thead.replaceChildren(head);
  tbody.replaceChildren(...table.map((row) => {
    const r = document.createElement('tr');
    r.append(el('td', `${row.w}x${row.h}`));
    for (const cell of row.cells) {
      const td = el('td', cell ? ms(cell.median) : '…');
      if (cell) td.append(el('span', ` (${ms(cell.min)}–${ms(cell.max)})`, 'hint'));
      r.append(td);
    }
    r.append(el('td', row.n == null ? '–' : String(row.n)));
    return r;
  }));
}

function markdown(cols, table) {
  const lines = [`| Size | ${cols.map(label).join(' | ')} |`, `|---|${cols.map(() => '---|').join('')}`];
  for (const row of table) lines.push(`| ${row.w}x${row.h} | ${row.cells.map((c) => (c ? ms(c.median) : '')).join(' | ')} |`);
  lines.push('', `ms per image, top-k ${TOP_K}, median of ${$('runs').value} runs after ${$('warmup').value} warm-up runs.`);
  lines.push(`${browserName()}; GPU: ${env.webgpu ? env.adapter : 'none'}; ${navigator.userAgentData?.platform || navigator.platform}.`);
  return lines.join('\n');
}

async function run() {
  const cols = columns();
  if (!cols.length) { $('progress').textContent = 'Choose at least one precision.'; return; }
  const runs = Math.max(1, +$('runs').value | 0), warmup = Math.max(0, +$('warmup').value | 0);
  const cpu = cols[0].backend === 'cpu';
  const sizes = cpu ? SIZES.slice(0, 1) : SIZES;
  stopping = false;
  $('start').disabled = true;
  $('stop').disabled = false;
  $('copy').disabled = true;
  const image = await loadImage('../samples/tum-1.jpg');
  const inputs = sizes.map(([w, h]) => pixels(image, w, h, true));
  const table = sizes.map(([w, h]) => ({ w, h, cells: cols.map(() => null), n: null }));
  render(cols, table);
  try {
    const bytes = await weights();
    for (let c = 0; c < cols.length && !stopping; c++) {
      $('progress').textContent = `${label(cols[c])}: compiling the shaders…`;
      const extractor = await jkp.create({ weights: bytes, backend: cols[c].backend, precision: cols[c].precision, topK: TOP_K });
      try {
        await extractor.extract(inputs[0]);                       // first run: shader compilation, buffers
        for (let s = 0; s < sizes.length && !stopping; s++) {
          const times = [];
          let n = 0;
          for (let k = 0; k < warmup + runs && !stopping; k++) {
            $('progress').textContent = `${label(cols[c])}, ${sizes[s][0]}x${sizes[s][1]}: ` +
              (k < warmup ? `warm-up ${k + 1} of ${warmup}` : `run ${k - warmup + 1} of ${runs}`);
            const t = performance.now();
            const r = await extractor.extract(inputs[s]);
            const dt = performance.now() - t;
            n = r.n;
            if (k >= warmup) times.push(dt);
          }
          if (times.length) {
            table[s].cells[c] = { median: median(times), min: Math.min(...times), max: Math.max(...times) };
            table[s].n = n;
          }
          render(cols, table);
        }
      } finally {
        extractor.destroy();
      }
    }
    $('progress').textContent = stopping ? 'Stopped.' : 'Done.';
  } catch (e) {
    console.error(e);
    $('progress').textContent = `Failed: ${e.message ?? e}`;
  }
  last = markdown(cols, table);
  $('md').textContent = last;
  $('copy').disabled = false;
  $('start').disabled = false;
  $('stop').disabled = true;
}

async function main() {
  env = await environment();
  renderEnv();
  const unsupported = unsupportedMessage(env);
  if (unsupported) {
    $('banner').hidden = false;
    $('banner').textContent = unsupported;
    return;
  }
  if (!env.webgpu) {
    $('banner').hidden = false;
    $('banner').textContent = 'WebGPU is not available in this browser; only the CPU backend can be timed.';
    document.querySelector('input[name="backend"][value="webgpu"]').disabled = true;
    document.querySelector('input[name="backend"][value="cpu"]').checked = true;
  }
  if (!env.f16) {
    $('fp16').checked = false;
    $('fp16').disabled = true;
    $('fp16-hint').textContent = env.webgpu ? '(this adapter has no shader-f16)' : '';
  }
  for (const r of document.querySelectorAll('input[name="backend"]')) {
    r.addEventListener('change', () => {
      const cpu = document.querySelector('input[name="backend"]:checked').value === 'cpu';
      $('fp16').disabled = cpu || !env.f16;
      if (cpu) { $('runs').value = Math.min(+$('runs').value, 5); $('warmup').value = Math.min(+$('warmup').value, 1); }
    });
  }
  $('start').addEventListener('click', run);
  $('stop').addEventListener('click', () => { stopping = true; });
  $('copy').addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(last); $('copy').textContent = 'Copied'; }
    catch { $('copy').textContent = 'Select the text and copy it'; }
    setTimeout(() => { $('copy').textContent = 'Copy as Markdown'; }, 1600);
  });
  $('progress').textContent = 'Loading the module…';
  await Promise.all([jkp.module(), weights()]);
  $('progress').textContent = 'Ready.';
  $('start').disabled = false;
}

main().catch((e) => {
  console.error(e);
  $('progress').textContent = `Could not start: ${e.message ?? e}`;
});
