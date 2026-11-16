# Dev tools

Python + PyTorch are used here only to export weights and validate against the original ALIKED.
They are never a runtime dependency of the library.

```sh
python -m venv .venv && .venv/bin/pip install -r tools/requirements.txt
tools/fetch_upstream.sh                      # original ALIKED at the pinned commit -> third_party/ALIKED
A=third_party/ALIKED/assets

python tools/export_weights.py               # -> weights/aliked-n16.jkpw (BatchNorm folded)
python tools/compare.py $A/tum/*.png         # original vs reference forward pass from the .jkpw file
python tools/compare.py $A/tum/*.png --a upstream:cpu --b upstream:cuda
python tools/dump_golden.py $A/tum/*.png     # per-layer test vectors -> build/golden/*.jkpt (not committed)
python tools/compare.py $A/tum/*.png --b "cmd:build/release/jkp extract"   # original vs the C++ library
python tools/compare.py $A/tum/*.png --b "cmd:build/release/jkp extract --backend cuda"
```

| File | Purpose |
|---|---|
| `jkpfile.py` | `.jkpw` / `.jkpt` container reader and writer; the format is documented at the top of the file |
| `upstream.py` | runs the original ALIKED on CPU or CUDA and exposes its intermediate tensors |
| `folded.py` | forward pass from a `.jkpw` file only; the executable specification for the C++ backends |
| `export_weights.py` | checkpoint -> `.jkpw` |
| `dump_golden.py` | golden test vectors from the original |
| `compare.py` | compares two implementations: `upstream`, `folded`, or `cmd:<program>` (the `jkp` CLI, any backend) |

The original needs a compiled CUDA extension (`custom_ops/get_patches`). When it is not built,
`upstream.py` substitutes an equivalent pure-torch version so the original runs on CPU-only machines.
