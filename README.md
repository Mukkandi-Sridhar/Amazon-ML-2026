# Amazon ML Challenge 2026: Business Entity Resolution

This is a solution for the Amazon ML Challenge 2026 on Unstop. Given business records from three noisy sources, find every Source-2 / Source-3 record that refers to the same real-world business as each Source-1 record.

- **Score:** macro-averaged F0.5, computed per Source-1 entity with singletons included.
- **Data:** train has 2.2M S1, 5.0M S2 and 5.3M S3 records. Test has 1.7M S1, 4.9M S2 and 5.1M S3, and adds a third country (France) that does not appear in train.

## Result

| | |
|---|---|
| Out-of-fold macro F0.5 on the **full** training set (all 2.2M S1 entities) | **0.9841** |
| Candidate recall ceiling (retrieval + cascade) | 97.7% of true pairs |

The full methodology write-up is in [`Documentation.md`](Documentation.md), which is the filled-in official template.

## Pipeline

```
TSV ─► normalise (names, addresses, Indic-script transliteration, phonetic skeletons)
    ─► retrieval: sparse TF-IDF, per country: name top-10 ∪ address top-10 ∪ combined top-15
    ─► stage 1: LightGBM cascade on retrieval scores → keep top-6 per query  (= candidate_pairs.tsv)
    ─► stage 2: LightGBM on 94 pair features (string, phonetic, house-number geometry)
    ─► stage 3: LightGBM re-scoring with query- and entity-level context
    ─► assignment: every S2/S3 record → its best S1 entity if p > τ (τ tuned on OOF macro F0.5)
```

## Reproduce

Requirements: Python 3.11, 4 CPU cores, 16 GB RAM, ~40 GB free disk. No GPU needed. The full run takes about 3–4 hours.

```bash
pip install -r requirements.txt
# dataset/ must contain train/ and test/ exactly as shipped in student_resource.zip
./run_pipeline.sh dataset work output
```

This writes `output/matching_results.tsv` (the file scored on the leaderboard) and `output/candidate_pairs.tsv`, then runs the official validator on both.

## Layout

| Path | Contents |
|---|---|
| `run_pipeline.sh` | end-to-end entry point |
| `src/prepare.py` | TSV → parquet |
| `src/normalize.py`, `src/run_normalize.py` | normalisation, transliteration, phonetic skeleton |
| `src/blocking.py`, `src/run_blocking_full.py` | three-channel sparse retrieval |
| `src/features.py` | retrieval, string-similarity and house-number features |
| `src/train_stage1.py`, `src/run_features.py` | stage-1 cascade and pair features (streamed in chunks) |
| `src/run_stage2.py` | cross-fitted stage-2 matcher |
| `src/run_stage3.py` | context model, threshold tuning, output writer |
| `src/metric.py` | exact macro F0.5 |
| `utils/validate_submission.py` | official format validator (unchanged) |

## Fair play

The solution uses only the provided data. It makes no external lookups, API calls or geocoding, and uses no pretrained models; every model is a LightGBM (MIT-licensed) trained from scratch on the challenge data. The only static tables are general-knowledge abbreviation lists (US and Indian state codes, street types, legal-entity suffixes).
