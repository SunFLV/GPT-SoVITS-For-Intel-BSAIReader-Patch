#!/usr/bin/env bash
# Apply the backend-agnostic, v5Turbo-ready BSAIReader patch to a pristine extraction.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-$PWD}"
PATCH="$HERE/bsaireader-gpt-sovits.patch"

if [ ! -f "$TARGET/GPT_SoVITS/TTS_infer_pack/TTS.py" ]; then
  echo "ERROR: $TARGET does not look like a GPT-SoVITS-For-Intel checkout." >&2
  exit 1
fi

cd "$TARGET"
git -c core.autocrlf=false apply --check "$PATCH"
git -c core.autocrlf=false apply "$PATCH"
echo "Applied: $PATCH"
echo "Optional runtime-state patch (not applied): $HERE/optional/weight.json.patch"
