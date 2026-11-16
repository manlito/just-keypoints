// Generated test images, shared by the tests that need no image files.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// Deterministic test picture: gradient background, rectangles, discs and a little noise.
inline std::vector<uint8_t> picture(int w, int h, int channels, int stride, uint32_t seed) {
  auto next = [&seed]() {
    seed = seed * 1664525u + 1013904223u;
    return seed >> 8;
  };
  std::vector<float> rgb(static_cast<size_t>(w) * h * 3);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      for (int c = 0; c < 3; ++c) rgb[(static_cast<size_t>(y) * w + x) * 3 + c] = 60.0f + 80.0f * (c == 0 ? x : y) / (c == 2 ? h : w);
  for (int shape = 0; shape < 40; ++shape) {
    int cx = next() % w, cy = next() % h, rx = 3 + next() % (w / 5 + 1), ry = 3 + next() % (h / 5 + 1);
    bool disc = next() % 3 == 0;
    float color[3] = {static_cast<float>(next() % 256), static_cast<float>(next() % 256), static_cast<float>(next() % 256)};
    for (int y = std::max(0, cy - ry); y < std::min(h, cy + ry); ++y)
      for (int x = std::max(0, cx - rx); x < std::min(w, cx + rx); ++x) {
        float dx = static_cast<float>(x - cx) / rx, dy = static_cast<float>(y - cy) / ry;
        if (disc && dx * dx + dy * dy > 1.0f) continue;
        for (int c = 0; c < 3; ++c) rgb[(static_cast<size_t>(y) * w + x) * 3 + c] = color[c];
      }
  }
  std::vector<uint8_t> out(static_cast<size_t>(stride) * h, 0xAB);  // row padding holds junk on purpose
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const float* p = &rgb[(static_cast<size_t>(y) * w + x) * 3];
      float noise = static_cast<float>(next() % 9) - 4.0f;
      uint8_t* o = &out[static_cast<size_t>(y) * stride + static_cast<size_t>(x) * channels];
      auto u8 = [&](float v) { return static_cast<uint8_t>(std::fmin(255.0f, std::fmax(0.0f, v + noise))); };
      if (channels == 1) o[0] = u8((p[0] + p[1] + p[2]) / 3.0f);
      else o[0] = u8(p[0]), o[1] = u8(p[1]), o[2] = u8(p[2]);
      if (channels == 4) o[3] = static_cast<uint8_t>(next() % 256);  // alpha must be ignored
    }
  return out;
}
