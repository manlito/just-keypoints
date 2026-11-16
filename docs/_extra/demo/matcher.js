// The demo's matcher, in a worker so the page stays responsive: mutual nearest neighbours on descriptor
// similarity, with Lowe's ratio test in both directions. It lives in the demo only; the library stops at
// keypoints and descriptors.
//
// Descriptors are L2-normalized, so the distance is sqrt(2 - 2 s) for a dot product s. The ratio test keeps a
// pair when d(best) < ratio * d(second best); ratio 1 keeps every mutual nearest neighbour.

self.onmessage = (event) => {
  const { a, b, na, nb, dim, ratio, id } = event.data;
  const t0 = performance.now();
  const bestA = new Int32Array(na).fill(-1), s1A = new Float32Array(na).fill(-2), s2A = new Float32Array(na).fill(-2);
  const bestB = new Int32Array(nb).fill(-1), s1B = new Float32Array(nb).fill(-2), s2B = new Float32Array(nb).fill(-2);
  for (let i = 0; i < na; i++) {
    const oi = i * dim;
    let best = -1, s1 = -2, s2 = -2;
    for (let j = 0; j < nb; j++) {
      const oj = j * dim;
      let s = 0;
      for (let k = 0; k < dim; k += 4) {
        s += a[oi + k] * b[oj + k] + a[oi + k + 1] * b[oj + k + 1] + a[oi + k + 2] * b[oj + k + 2] + a[oi + k + 3] * b[oj + k + 3];
      }
      if (s > s1) { s2 = s1; s1 = s; best = j; } else if (s > s2) s2 = s;
      if (s > s1B[j]) { s2B[j] = s1B[j]; s1B[j] = s; bestB[j] = i; } else if (s > s2B[j]) s2B[j] = s;
    }
    bestA[i] = best; s1A[i] = s1; s2A[i] = s2;
  }
  const dist = (s) => Math.sqrt(Math.max(0, 2 - 2 * s));
  const pass = (s1, s2) => ratio >= 1 || s2 <= -2 || dist(s1) < ratio * dist(s2);
  const pairs = [], sims = [];
  for (let i = 0; i < na; i++) {
    const j = bestA[i];
    if (j < 0 || bestB[j] !== i) continue;
    if (!pass(s1A[i], s2A[i]) || !pass(s1B[j], s2B[j])) continue;
    pairs.push(i, j);
    sims.push(s1A[i]);
  }
  const result = { id, pairs: Int32Array.from(pairs), sims: Float32Array.from(sims), ms: performance.now() - t0 };
  self.postMessage(result, [result.pairs.buffer, result.sims.buffer]);
};
