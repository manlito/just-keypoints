#!/usr/bin/env bash
# Fetch the original implementations at their pinned commits into third_party/:
# ALIKED (code and checkpoints), RaCo (code, plus the released checkpoint as third_party/raco.pth) and XFeat
# (verlab/accelerated_features: code and its checkpoints, weights/xfeat.pt and weights/xfeat-lighterglue.pt).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

fetch() {  # name url commit
  local dest="$ROOT/third_party/$1"
  if [ ! -d "$dest/.git" ]; then
    mkdir -p "$(dirname "$dest")"
    git clone "$2" "$dest"
  fi
  git -C "$dest" fetch -q origin "$3" 2>/dev/null || git -C "$dest" fetch -q origin
  git -C "$dest" checkout -q "$3"
  echo "$1 at $(git -C "$dest" rev-parse HEAD) in $dest"
}

fetch ALIKED https://github.com/Shiaoming/ALIKED.git 683d7c65197395c0b3f01ebe76e1084a27e73a65
fetch RaCo https://github.com/cvg/RaCo.git ee06fa68dbb20303f334fbc257dad71d4149603b
fetch accelerated_features https://github.com/verlab/accelerated_features.git e92685f57f8318b18725c5c8c0bd28c7fe188d9a

RACO_PTH="$ROOT/third_party/raco.pth"
RACO_SHA256=79f0d862bb41abedd8dda33a9b0120f4dcd619c43a90398175bad1a019192e3a
if [ ! -f "$RACO_PTH" ]; then
  curl -fL -o "$RACO_PTH.part" https://github.com/cvg/RaCo/releases/download/v1.0.0/raco.pth
  mv "$RACO_PTH.part" "$RACO_PTH"
fi
echo "$RACO_SHA256  $RACO_PTH" | shasum -a 256 -c -

XFEAT_SHA256=0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b
echo "$XFEAT_SHA256  $ROOT/third_party/accelerated_features/weights/xfeat.pt" | shasum -a 256 -c -
