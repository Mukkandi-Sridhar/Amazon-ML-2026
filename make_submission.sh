#!/usr/bin/env bash
# Build <team_name>_submission.zip in the layout required by the challenge.
# Usage: ./make_submission.sh <team_name> <output_dir_with_tsvs> <dest_dir>
set -euo pipefail
TEAM=${1:?team name}
OUT=${2:-output}
DEST=${3:-.}
ROOT=$(cd "$(dirname "$0")" && pwd)
STAGE=$(mktemp -d)
PKG="$STAGE/${TEAM}_submission"
mkdir -p "$PKG/output" "$PKG/code/business_entity_resolution/src"
cp "$OUT/matching_results.tsv" "$OUT/candidate_pairs.tsv" "$PKG/output/"
cp "$ROOT"/src/*.py "$PKG/code/business_entity_resolution/src/"
cp "$ROOT/run_pipeline.sh" "$PKG/code/business_entity_resolution/"
cp "$ROOT/README.md" "$ROOT/requirements.txt" "$PKG/code/business_entity_resolution/"
mkdir -p "$PKG/code/business_entity_resolution/utils"
cp "$ROOT/utils/validate_submission.py" "$PKG/code/business_entity_resolution/utils/"
cp "$ROOT/Documentation.md" "$PKG/Documentation_template.md"
(cd "$PKG" && zip -q -r "$DEST/${TEAM}_submission.zip" .)
rm -rf "$STAGE"
echo "wrote $DEST/${TEAM}_submission.zip"
