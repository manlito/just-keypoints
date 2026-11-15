#!/usr/bin/env bash
# Fetch the original ALIKED repository at the pinned commit into third_party/ALIKED.
set -euo pipefail
COMMIT=683d7c65197395c0b3f01ebe76e1084a27e73a65
DEST="$(cd "$(dirname "$0")/.." && pwd)/third_party/ALIKED"
if [ ! -d "$DEST/.git" ]; then
  mkdir -p "$(dirname "$DEST")"
  git clone https://github.com/Shiaoming/ALIKED.git "$DEST"
fi
git -C "$DEST" fetch -q origin "$COMMIT" 2>/dev/null || git -C "$DEST" fetch -q origin
git -C "$DEST" checkout -q "$COMMIT"
echo "ALIKED at $(git -C "$DEST" rev-parse HEAD) in $DEST"
