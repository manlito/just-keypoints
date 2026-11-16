"""Test data for the browser test page (web/test.html): images as raw RGBA and the native CPU backend's
results on them, in threshold mode and top-k 1000.

    python tools/web_testdata.py --jkp build/release/jkp --out build/wasm/web/testdata IMAGE...

Each image is decoded once here (PIL), written as a lossless PNG for the native `jkp extract` and as raw
RGBA (u32 width, u32 height, then the pixels) for the page, so both see exactly the same pixels whatever
the browser's JPEG decoder does. Writes manifest.json next to them.
"""
import argparse
import json
import os
import struct
import subprocess
import sys

from PIL import Image

MODES = {"threshold": [], "topk": ["--top-k", "1000"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jkp", required=True, help="native jkp command line tool")
    ap.add_argument("--out", required=True, help="output directory (served next to test.html)")
    ap.add_argument("--weights", help=".jkpw for the native run (default: embedded)")
    ap.add_argument("images", nargs="+")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    entries = []
    for i, path in enumerate(args.images):
        rgb = Image.open(path).convert("RGB")
        name = f"{i:02d}_{os.path.splitext(os.path.basename(path))[0]}"
        png = os.path.join(args.out, name + ".png")
        rgb.save(png)
        with open(os.path.join(args.out, name + ".rgba"), "wb") as f:
            f.write(struct.pack("<II", rgb.width, rgb.height))
            f.write(rgb.convert("RGBA").tobytes())
        refs = {}
        for mode, flags in MODES.items():
            ref = f"{name}.{mode}.jkpt"
            cmd = [args.jkp, "extract", png, "--backend", "cpu", "--out", os.path.join(args.out, ref)] + flags
            if args.weights:
                cmd += ["--weights", args.weights]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            refs[mode] = ref
        os.remove(png)
        entries.append({"name": name, "source": os.path.basename(path), "width": rgb.width, "height": rgb.height,
                        "pixels": name + ".rgba", "refs": refs})
        print(f"{name}: {rgb.width}x{rgb.height}", file=sys.stderr)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"modes": {"threshold": {"topK": -1}, "topk": {"topK": 1000}}, "images": entries}, f, indent=1)


if __name__ == "__main__":
    main()
