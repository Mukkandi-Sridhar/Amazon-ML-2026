#!/usr/bin/env bash
# End-to-end: data -> normalisation -> blocking -> stage-1 cascade -> features
#             -> stage-2 matcher -> stage-3 context model -> outputs
# Usage: ./run_pipeline.sh <dataset_dir (with train/ and test/)> <work_dir> <output_dir>
set -euo pipefail
DATA=${1:-dataset}
WORK=${2:-work}
OUT=${3:-output}
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -f "$HERE/normalize.py" ]; then SRC="$HERE"; else SRC="$HERE/src"; fi
DATA=$(cd "$DATA" && pwd); mkdir -p "$WORK" "$OUT"; WORK=$(cd "$WORK" && pwd); OUT=$(cd "$OUT" && pwd)
cd "$SRC"

python prepare.py "$DATA" "$WORK"                       # TSV -> parquet
python run_normalize.py "$WORK"                         # normalised name/address fields
python run_blocking_full.py "$WORK" train               # candidate generation (train)
python run_blocking_full.py "$WORK" test                # candidate generation (test)
python train_stage1.py "$WORK" "$WORK/train_gt.parquet" "$WORK/stage1_light.txt"
python run_features.py "$WORK" train "$WORK/stage1_light.txt"
python run_features.py "$WORK" test "$WORK/stage1_light.txt"
python run_stage2.py "$WORK" "$WORK/train_gt.parquet" all
python run_stage3.py tune "$WORK" "$WORK/train_gt.parquet"
python run_stage3.py write "$WORK" test "$OUT" "$DATA/test/test_source1.tsv"      # also writes candidate_pairs.tsv
# final decision: expected-F0.5 + test-specific twin gate (house-number offsets), overwrites matching_results.tsv
python run_final2.py "$WORK" "$WORK/train_oof.parquet" "$WORK/test_scores.parquet" p3 "$OUT" "$DATA/test/test_source1.tsv" 0.9
python "$SRC/../utils/validate_submission.py" --matching "$OUT/matching_results.tsv" \
       --candidate "$OUT/candidate_pairs.tsv" --test-dir "$DATA/test"
