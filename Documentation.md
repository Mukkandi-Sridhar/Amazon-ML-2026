# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We treat every Source-2 / Source-3 record as a *query* that belongs to **at most one** Source-1 entity (a property that holds exactly in the training labels: 7,638,365 matched IDs, all unique). A three-channel sparse TF-IDF retrieval (name, address, and both combined) generates candidates. A cheap LightGBM cascade keeps the 6 best per query. A 94-feature LightGBM matcher scores them, and a context model re-scores each pair using competing claims on the same query and the same Source-1 entity. Each query is assigned to its single best Source-1 entity only when the calibrated probability clears a threshold tuned directly for macro F0.5. Key ideas that go beyond generic string similarity are a coarse **phonetic consonant skeleton** that matches names written in 7 Indic scripts against their English originals, and **house-number geometry features** that separate "neighbouring business" hard negatives from typo'd true matches.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA findings on the 2.2M / 5.0M / 5.3M training records (S1 / S2 / S3):

| Finding | Consequence |
|---|---|
| Every S2/S3 record matches **at most one** S1 entity; 26% of S2/S3 records match none. | Solve it as an *assignment*: each query picks its best S1 or abstains. This is a strong precision constraint. |
| Matches always share the same `country` label. | Retrieval is run per country label. The label set is handled as an open set, so France (test only) works unchanged. |
| 5.6% of S1 entities are singletons; clusters hold 1–11 records (mean 3.46). | False merges on singletons cost a full 1.0, so abstaining must be cheap. |
| 23.5% of India S2 names and 13% of S3 names are written in Indic scripts (Devanagari, Bengali, Telugu, Tamil, Gujarati, Gurmukhi, Malayalam); Indic state names also appear inside addresses. | Transliterate to ASCII, then compare *phonetic skeletons* rather than spellings. |
| Name noise: legal-suffix shuffles (`LLC Beacon…`, `Pvt. … Ltd.`), abbreviations, typos (`Heart1and`, `Hfiglcahdn`), duplicated words, trailing filler (`Services`, `Partners`), bracketed legal forms (`[Corp]`), phone numbers, `| www.site.com` tails, `M/s`, `>>`/`--` prefixes, DBA (`X doing business as Y`), and names replaced by a **domain** (`beaconbiotechnologies.com`). | Dedicated normaliser: legal forms canonicalised and split off, DBA split, domain stem extracted, concatenated-name skeleton. |
| Address noise: component reordering, abbreviations (`Ct`/`Court`, `R.`/`Rue`), state code ↔ name, city variants (`Richmond City`, `City Of Superior`, `Chna Grove`), `PO Box`/`PMB`/`null`/`N/A` insertions, leading zeros (`00380`), unit suffixes (`9616-C`), missing address (3.4%). | Order-free token bags, number extraction with zero stripping, state canonicalisation, filler removal. |
| **Hard negatives**: neighbouring businesses whose house number differs by a small amount in the *last* digit (`7344` vs `7342 Lambton Green`), while true matches show typo-style digit errors, often in the *leading* digit (`509` vs `409 Hickory Lane`). | House-number geometry features (numeric difference, which digit differs). |

**Transliteration details that mattered:**
- Malayalam chillu letters are expanded before `unidecode`.
- Tamil `ச` is rendered as `s` (it is `c` in `unidecode` but pronounced "s", as in "South" and "Consulting").
- The skeleton merges voiced/unvoiced consonants, drops vowels, `h`, `y` and final `-g` (so "-ing" = "-in"), and treats soft `c`/`g` and silent `gh` as English does.
- Transliterated legal words ("limittett", "praaivett", "elelpi") are recognised through their skeletons.

With these rules, e.g., `ராயல் ஈஸ்டர்ன் டிரேடிங் பிரைவேட் லிமிடெட்` and "Royal Eastern Trading Private Limited" both reduce to `rl strn trtn` + legal `ltd pvt`.

### 2.2 Solution Strategy

**Approach Type:** Blocking (sparse retrieval) → 3-stage LightGBM cascade → constrained assignment
**Core Innovation:** Treating matching as a many-to-one assignment with context re-scoring, plus a script-agnostic phonetic skeleton and house-number geometry features.

```
TSV ─► normalise (names, addresses, transliteration, skeletons)
    ─► retrieval: name top-10 ∪ address top-10 ∪ combined top-15   (~23 cands / query)
    ─► stage 1: LightGBM on 23 retrieval features → keep top-6       (candidate_pairs.tsv)
    ─► stage 2: LightGBM on 94 pair features (string, phonetic, numeric)
    ─► stage 3: LightGBM on stage-2 score + query/entity context
    ─► assignment: each query → argmax S1 if p > τ   (τ tuned for macro F0.5)
```

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** two IDF-weighted sparse token channels per record (IDF from Source 1 of the same country; tokens occurring in more than 1% of S1 records are dropped):
  - *Name channel:* normalised core-name words (legal forms removed), phonetic skeleton of every word, skeleton of the whole concatenated name (so `beaconbiotechnologies.com` hits `Beacon Biotechnologies`), and DBA alias words.
  - *Address channel:* address words (abbreviations expanded, states and filler removed) and every number in the address (leading zeros stripped).
  - *Address bigrams* (consecutive normalised address words such as `main_cross`, `j_p`, `p_nagar`). Indian addresses are built from very common words that the document-frequency cap removes individually; as pairs they become specific.
  - Three cosine top-K searches (`sparse_dot_topn`, per country): name-only top-10, address-only top-10, and combined (average) top-15.
  - **Exact-key passes** (hash joins) that rescue records whose name is shared by dozens of S1 entities, or whose address is truncated to a city and a number:
    - (core name, house number), (phonetic skeleton, house number) and (concatenated skeleton, house number), used when the key is shared by ≤50 S1 records;
    - exact core name, skeleton or concatenated skeleton alone, used when shared by ≤10 S1 records.

    Which key matched is passed on as a flag.
  - The union gives ≈24 candidates per query. On a 15% query sample, adding bigrams and keys raised union recall from 97.57% to **97.88%** (Indic-script India records: 92.9% → **95.3%**).
- **Stage-1 cascade:** a small LightGBM (63 leaves, 150 trees) over 30 cheap features, including the exact-key flags (both channel cosines, per-channel ranks, gap to the query's best, margin to the runner-up, query flags). It keeps the **top 6** candidates per query. This final set is what the matching models see, and it is written to `candidate_pairs.tsv`.
- **Candidate set = final model input.** Candidate generation is a cascade, and `candidate_pairs.tsv` is its *last* stage, i.e. exactly the pairs the final (stage-3) model scores:
  1. retrieval: three TF-IDF top-K passes plus exact keys, ≈24 candidates per S2/S3 record;
  2. cascade filter 1: cheap LightGBM on retrieval scores, top-8 per record;
  3. cascade filter 2: pairwise LightGBM, stage-2 probability ≥ 0.05.

  The result is **≈3.6 candidates per Source-1 entity** on test (v4: 3.87 with cut-off 0.01). This is a reduction ratio of > 99.9999% versus all same-country pairs, while keeping **97.45% of all true train pairs**. Raising the cut-off from 0.01 to 0.05 shrinks the set by 7% with no change in OOF macro F0.5 (0.98412 → 0.98413): pairs below it are essentially never accepted.
- **How true matches were not lost** (measured on train):
  - Retrieval union recall is **97.57%** of all true pairs (99.74% for US records that have an address).
  - The cascade keeps **99.41%** of the retrieved true pairs in the top 6, for an overall recall ceiling of **≈97.0%**.
  - Most of the unrecoverable pairs have *no address* and a generic name that several S1 entities share, so they are ambiguous by construction. F0.5 would not reward guessing on them anyway.

---

## 4. Matching Model

**Features used (stage 2, 94 in total):**
- **Name:** RapidFuzz `ratio`, `token_sort_ratio`, `token_set_ratio`, `partial_ratio`, Jaro-Winkler on the core name; ratio / token-set on the phonetic skeleton; ratio / partial ratio on the concatenated skeleton (domain names); token-set on the full name including legal forms; DBA alias vs core; shared-word counts and containment in both directions; first-word equality; legal-form agreement (`llc`↔`llc`, `pvt ltd`↔`pvt ltd`, …).
- **Address:** token-set / sort / ratio / partial-token-set on normalised address words; shared-word counts and containment; number-set token overlap, shared numbers and containment; primary house-number equality and Levenshtein distance; state equality after canonicalisation; city ratio / partial ratio.
- **Token differences (16):** for phonetic name tokens and address words, how many tokens appear on only one side, and their summed and maximum IDF (from Source 1), plus the IDF of the shared tokens. A rare differing word ("Consultants" vs "Solutions") marks a different business, while common filler ("Services", "Center") or typos mark noise. Legal forms of both sides are passed as LightGBM categorical codes, so the model learns which legal-form changes are formatting noise (LLC ↔ L.L.C.) and which mark a twin business (Corp → Inc).
- **House-number geometry (14):** absolute, log and relative difference; same length; Hamming distance on right-aligned digits; lowest and highest differing digit position; whether the first and last digits agree; whether one side's house number appears anywhere in the other's numbers; minimum absolute distance to any number on the other side.
- **Retrieval / cascade context:** name and address cosines, their ranks in each search, the query's best cosines, gaps and margins, the stage-1 probability, its rank, and the query's max and sum.
- **Query flags:** address missing, name in a non-Latin script, name is a domain, source (S2/S3), token counts.

`country` is deliberately **not** a feature, so France is scored with the same country-agnostic model.

**Name-ambiguity features (stage 2):** how many S1 records in the same country share the query's (and the candidate's) exact core name and phonetic skeleton. A unique name makes a name-only match safe; "Midwest Coalition" occurs 21 times in train S1.

**Stage-3 context features:**
- *Query level:* stage-2 probability; its rank within the query; the query's max, sum and margin over its runner-up; the number of candidates.
- *Entity level (excluding the current pair):* sum and max of the other pairs' probabilities; this pair's rank among them; how many *other* queries have this entity as their confident top choice (overall and from the same source); their mean confidence.
- *Claimant consensus:* how many other claimants of the same S1 entity share this query's house number and core name, raw and probability-weighted; whether the query carries the entity's majority (probability-weighted) house number and name; whether that majority agrees with Source 1.

  Why this matters: when several independent S2/S3 records agree on a value that differs from Source 1, it is usually Source 1 that carries the typo. Among true matches whose house number disagrees with S1, 45% have another claimant with the same number, versus 7.7% for false merges. Without these features, a context model penalises every 5th or 6th claimant of an entity (a cluster-size prior) even when all of them agree perfectly.

**Model type:** LightGBM gradient-boosted trees (binary log-loss) at every stage:
- Stage 2: 255 leaves, 500 rounds, learning rate 0.1.
- Stage 3: 127 leaves, 300 rounds.

Stages 2 and 3 are **2-fold cross-fitted** by query on train. Every training pair gets an out-of-fold probability, and the two fold models are averaged for test.

**Decision rule / threshold selection:** each query is assigned to its highest-scoring candidate. Two rules are evaluated with the **exact competition metric** (macro F0.5 over *all* 2.2M train S1 entities, singletons included) on out-of-fold predictions, and the better one is used:
1. A global threshold τ on p₃, chosen by grid search.
2. Per-entity **expected-F0.5 maximisation.** Each entity's claimants are sorted by p₃, and the prefix length k maximising 1.25·Σ_{i≤k}p_i / (0.25·(Σ_i p_i + c) + k) is kept. k = 0 ("no match") is chosen when P(no true match) = Π(1−p_i)·e^{−c} is larger. The constant c is tuned on OOF.

### 4.1 Train → test distribution shift and label-shift calibration

The test set is **not** distributed like train:
- It has 5.8 S2/S3 records per S1 entity versus 4.7 in train.
- The share of records whose best candidate scores between 0.01 and 0.8 triples (14% vs 4.7%) in *every* country, including France.

The extra records are mostly **twin businesses**: the same street with a house number a few units away, often with one name word or the legal form changed. A model calibrated on train therefore overstates match probabilities on test. The first leaderboard submissions confirmed it: 0.966 / 0.962 against out-of-fold 0.977 / 0.979, and the more aggressive model scored *lower*.

We correct this without test labels, assuming only that true matches have the same score distribution on test as on train (label shift):
1. Estimate the number of true matches on test from the near-certain band (score > 0.999, 99.96% precise on train): N_pos,test = n_test(top) · prec_train(top) / P(top | match)_train. This gives **3.40 matches per S1 entity**, consistent with train's 3.46 and an independent sanity check.
2. For every score band, E[#matches on test] = N_pos,test · P(band | match)_train, so the test precision of the band is that count divided by n_test(band). The result is made monotone (isotonic).

   Example: the band 0.90–0.95 is 93% precise on train but only **54%** on test.
3. Each query's best-candidate score is replaced by its calibrated test precision, and the per-entity expected-F0.5 rule is applied to those probabilities.
4. The rule is chosen by **simulating** the test macro F0.5, drawing labels from the calibrated probabilities. On train the simulator reproduces the true OOF score (0.977 vs 0.979). On test it favours the calibrated expected-F rule over the train-tuned threshold by +0.011.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, out-of-fold on the full training set):** **0.9841** (stage-2 score only: 0.9827; threshold τ = expected-F0.5 rule (≈0.7) + twin gate)
- **Common false positives (wrong merges):**
  - Neighbouring businesses: same name, same street, house number off by a few units (`7344` vs `7342 Lambton Green`, `10204` vs `10203 Drew Hill Lane`). The geometry features were added for exactly this case.
  - Sibling businesses at the *same* address with a different trade word (`Jan Consultants Ltd` vs `Jan Solutions Ltd`, `Salasar Services` vs `Salasar Finance`).
- **Common false negatives (missed matches):**
  - Records with *no address* and a generic name that several S1 entities share (`Primary Care Medicine`, `Wildlife Center`).
  - A trade name entirely different from the S1 legal name at the same address (`Jaxiri Labs` vs `Parshwanath Publication Pvt Ltd`).
  - Heavily truncated Indian addresses combined with an Indic-script name.

---

### Leaderboard history (public split)

| Version | Change | OOF | Public LB |
|---|---|---|---|
| v1 | 3-stage cascade, threshold 0.7 | 0.9765 | 0.966 |
| v2 | + more data, claimant-consensus context | 0.9793 | 0.962 |
| v2-cal | v2 + band-level label-shift calibration | – | 0.964 |
| v3 | corrected transliteration, address bigrams, exact keys, token-difference features, **no** consensus, twin gate | 0.9830 | – |
| **v4** | v3 + house-number×address-word retrieval tokens (trade-name aliases), top-8 cascade | **0.9841** | **0.975** |
| **v5** | v4 + French article stop words, 2-model stage-2 ensemble, cascade cut-off 0.05 (3.66 candidates / S1) | **0.9842** | **0.975** |
| **v6 (final)** | 50/50 blend of v5 and v4 stage-3 scores on v5's candidate set (`src/blend_final.py`), same twin gate | **0.9844** | submitted |

What we learned from the leaderboard:
- Claimant consensus accepted test's twin groups.
- Band-level calibration was too pessimistic: test's genuine matches are also noisier, so mid-score pairs are mostly real.
- The only test-specific distractor with a clear fingerprint is the house-number-offset twin (offsets 1,2,3,4,5,7,9,11,13,21), which the gate handles.

## 6. Conclusion

A retrieval + cascade + context design scales to 26M records on a 4-core / 16 GB machine without a GPU, and it reaches **0.9841** macro F0.5 out-of-fold. The biggest wins came from framing the task as a many-to-one assignment, matching phonetic skeletons across scripts, and modelling *how* numbers differ rather than *whether* they differ. With more compute, the next steps would be a small (≤8B, Apache-2.0) multilingual cross-encoder for the ambiguous tail, and cross-source (S2↔S3) cluster consistency features.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:

| File | Role |
|---|---|
| `run_pipeline.sh` | End-to-end entry point: `./run_pipeline.sh <dataset_dir> <work_dir> <output_dir>` |
| `src/prepare.py` | TSV → parquet (explicit tab separator, no quoting) |
| `src/normalize.py`, `src/run_normalize.py` | Name/address normalisation, Indic transliteration, phonetic skeleton |
| `src/blocking.py`, `src/run_blocking_full.py` | Three-channel sparse TF-IDF retrieval per country |
| `src/features.py` | Stage-1 retrieval features, string-similarity features, house-number geometry |
| `src/train_stage1.py`, `src/run_features.py` | Stage-1 cascade (top-6) + pair features, streamed in chunks |
| `src/run_stage2.py` | Cross-fitted stage-2 matcher |
| `src/run_stage3.py` | Context model, threshold tuning on OOF macro F0.5, output writer |
| `src/metric.py` | Exact macro F0.5 implementation |

No external data, APIs, geocoders or pretrained models are used. The only dictionaries are general-knowledge abbreviation tables (US/Indian state codes, street types, legal forms).

### B. Additional Results

| Stage | Metric (train) |
|---|---|
| Retrieval union recall | 97.57% |
| Recall by segment: US / India-Latin / India-Indic-script / no-address | 99.74% / 97.61% / 92.93% / ≈77% |
| Stage-1 top-6 recall (of retrieved) | 99.41% |
| Overall candidate recall ceiling | ≈97.0% |
| Stage-2 pair precision / recall @0.5 (held-out sample) | 98.9% / 94.6% |
| Macro F0.5 OOF (final) | 0.9841 |
