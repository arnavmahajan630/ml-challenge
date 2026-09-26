# PLAN.md — Business Entity Resolution: Agent Build Spec

> **Audience: the coding agent only.** Build the complete pipeline described here in one pass, then run the experiment ladder (§12) with the keep/drop rules given. Work is organized in stages; each stage has a **Definition of Done (DoD)**. After every stage, append results to `experiments.csv` and update `reports/STATUS.md` (humans read this file; keep it current and short).

---

## 0. Objective and scoring (read first)

For every Source 1 (S1) record, output the list of Source 2/Source 3 (S2/S3) record IDs that are the same real-world business. S1 is deduplicated. An S1 can match 0..N records.

**Metric: macro F0.5 per S1 entity, averaged over all S1.**
| Ground truth | Prediction | Score |
|---|---|---|
| empty | empty | 1.0 |
| empty | non-empty | 0.0 |
| non-empty | empty | 0.0 |
| non-empty | non-empty | 1.25·P·R / (0.25·P + R) |

The winning levers, in order of expected impact:
1. **Per-entity decoding that maximizes expected F0.5** (§8). This is where most of the score is won or lost.
2. **Calibrated, context-aware match probabilities** (stacking of string features + fine-tuned neural scorers + group features).
3. **Blocking recall** (ceiling on everything).
4. **Robustness to the unseen country (France)** (the private LB will contain it).

---

## 1. Non-negotiable rules

1. **No external lookups.** No geocoding, registries, web APIs, scraped or external datasets. Only the provided files and pretrained open-weight models.
2. **Every model used** (blockers, scorers, rerankers) must be **MIT or Apache-2.0 licensed and ≤ 8B parameters.** Before using any model, read its HF model card, confirm license and parameter count, and record both in `reports/models.md`. If uncertain, don't use it.
3. **Country is an open set.** Test contains France, which is not in train. Never write logic like `if country in {"US","India"}`, never one-hot country, never filter records by country. Partition by the country *value* if needed, generically. Before final run: `grep -rniE "\"(us|india)\"" src/` must return nothing outside tests.
4. **Never use `entity_id` numeric values, file row order, or any ID-derived signal as a feature.** The package will be reviewed; ID leakage risks disqualification.
5. **TSV I/O only.** Always `pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)`. Write with `sep="\t"`, no index, no quoting. Empty ID list = empty string.
6. **No leakage.** Any learned score that becomes a feature for a downstream model must be produced **out-of-fold (OOF)** on train. Unsupervised statistics (IDF, TF-IDF vocab, token frequencies) may be fit on all text from train+test (no labels involved) and this must be documented.
7. **Reproducibility.** Global `SEED=42` for python/numpy/torch/lightgbm/sklearn; `torch.backends.cudnn.deterministic=True`; all behavior driven by YAML config; every artifact keyed by config hash; pinned `requirements.txt`.
8. **Outputs:** `matching_results.tsv` (`source1_entity_id`, `matched_entity_ids`) and `candidate_pairs.tsv` (`source1_entity_id`, `candidate_entity_ids`). One row per test S1, S2-/S3- IDs only, no duplicates, every ID exists in test, **matches ⊆ candidates**. `candidate_pairs.tsv` must be **exactly the set of pairs the final scoring model ran inference on**. Run `utils/validate_submission.py` after every write; it must print PASS.

---

## 2. Environment (AWS SageMaker)

- Dev + CPU stages + small/medium GPU models: SageMaker Studio JupyterLab space on `ml.g5.2xlarge` (1× A10G 24 GB).
- 7–8B LoRA stage (§7.3): `ml.g6e.xlarge` (1× L40S 48 GB) or `ml.g5.12xlarge` (4× A10G). Run as a SageMaker Training Job via the HF estimator or directly in a second Studio space. Code must work in both (read paths from config/env, not hard-coded).
- All artifacts sync to `s3://<bucket>/ber/<exp_id>/` via `src/utils/s3.py`. Local cache under `artifacts/`.
- Use bf16 on A10G/L40S. Keep batch sizes configurable.

---

## 3. Repository layout

```
code/business_entity_resolution/
├── src/
│   ├── cli.py                 # entrypoint: python -m src.cli <stage|all> --config ...
│   ├── config.py              # load YAML, compute config hash, seed everything
│   ├── data/
│   │   ├── io.py              # safe TSV read/write
│   │   ├── folds.py           # GroupKFold over S1, LOCO splits
│   │   └── eda.py             # writes reports/eda.md
│   ├── text/
│   │   ├── normalize.py       # normalization + token/digit extraction
│   │   └── variants.py        # mines abbreviation/variant pairs from train GT
│   ├── blocking/
│   │   ├── tfidf.py           # char n-gram retrieval
│   │   ├── dense.py           # bi-encoder retrieval (FAISS)
│   │   ├── tokens.py          # rare-token / postcode blocking
│   │   ├── biencoder_train.py # contrastive fine-tune (OOF folds)
│   │   └── union.py           # union, bidirectional, cap, prefilter
│   ├── features/
│   │   ├── pair.py            # pairwise string/number features
│   │   ├── context.py         # rank, gap, mutual-best, frequency features
│   │   └── group.py           # 2nd-stage group-consistency features
│   ├── models/
│   │   ├── gbm.py             # LightGBM OOF trainer/predictor
│   │   ├── cross_encoder.py   # fine-tune + OOF + test inference
│   │   ├── llm_reranker.py    # LoRA ≤8B pairwise classifier (optional)
│   │   ├── calibrate.py       # isotonic / temperature
│   │   └── stack.py           # level-2 stacker
│   ├── decode/
│   │   ├── assign.py          # 1-to-1 conflict resolution across S1
│   │   └── expected_f.py      # per-S1 expected-F0.5 set selection
│   ├── eval/
│   │   ├── metric.py          # exact macro F0.5
│   │   └── report.py          # breakdowns, error dumps
│   ├── submit/
│   │   └── write.py           # writes both TSVs + assertions + runs validator
│   └── utils/ (seed.py, s3.py, cache.py, log.py, timer.py)
├── configs/
│   ├── base.yaml
│   ├── abbrev.yaml            # abbreviation/legal-form maps
│   └── final.yaml             # frozen winning config (written at the end)
├── tests/                     # pytest: metric, io, normalize, decode, submission
├── README.md
└── requirements.txt
experiments.csv
reports/ (STATUS.md, eda.md, models.md, errors/, plots/)
output/ (matching_results.tsv, candidate_pairs.tsv)
```

Stage outputs are parquet files in `artifacts/<stage>/<config_hash>/`. Every stage reads its inputs from cache, so any stage can be re-run alone. `python -m src.cli all --config configs/final.yaml` must regenerate both output TSVs from raw data.

---

## 4. Stage S0 — Data audit, folds, metric

1. Load all 7 files; assert unique `entity_id` per file and correct prefixes.
2. `reports/eda.md` must answer:
   - row counts, empty-field rates, country distribution per source (train and test);
   - singleton rate in train S1; distribution of match-list size; S2 vs S3 share;
   - **does any S2/S3 ID appear in more than one GT list?** (decides 1-to-1 constraint);
   - are all GT pairs same-country? (decides country partitioning in blocking);
   - fraction of S2/S3 records matching nothing (distractor rate);
   - how often multiple S2 (or S3) records match the same S1 (intra-source duplicates);
   - 50 sampled positive pairs and 50 hard negatives per country, printed side by side;
   - frequency of repeated `name_core` across S1 (chain/franchise risk).
3. **Folds:** `GroupKFold(5)` over train S1 IDs, stratified by country × singleton flag (use `StratifiedGroupKFold`). Every S2/S3 record belongs to the fold of its GT S1; unmatched S2/S3 records are in the candidate pool for **all** folds.
4. **LOCO splits (France proxy):** train-on-US → eval-India and train-on-India → eval-US. `loco_f05` = mean of the two.
5. `eval/metric.py`: exact metric from §0. Tests: PS example → 0.714 (±0.001); all four empty/non-empty cases; duplicate IDs in a prediction are deduplicated before scoring.

**DoD:** tests pass; eda.md complete; fold assignment saved.

---

## 5. Stage S1 — Normalization

Per record, produce: `name_norm`, `name_core`, `name_legal` (set of legal-form tokens), `addr_norm`, `name_tokens`, `addr_tokens`, `digits` (all digit runs), `postcode_like` (digit runs of length 4–6, plus alphanumeric postal patterns), `house_no` (first short digit run in address), `landmark_flag`, `dba_flag`, `alias_names` (split on dba / d.b.a / t/a / trading as / "(...)").

Rules:
- Unicode NFKD → drop combining marks (stdlib `unicodedata`; avoid GPL libraries). Lowercase. Normalize quotes/dashes.
- `&`→`and`, `+`→`and`; remove punctuation except inside digit runs and between letters in abbreviations (`p.v.t.` → `pvt`); collapse whitespace.
- Apply `configs/abbrev.yaml` in both directions to a canonical form. Include generic multilingual legal forms and address words (corp/corporation, pvt/private, ltd/limited, co/company, inc, llc, llp, plc, intl, mfg, bros, sa, sas, sarl, eurl, gmbh; rd, st, ave/av, blvd/bd, ln, dr, nr/near, opp/opposite, bldg, flr/fl, apt, ste, sq, pl/place, ch/chemin, imp/impasse, fbg/faubourg, qtr/quartier). Treat `st` as ambiguous (street vs saint): keep both expansions as alternative tokens rather than forcing one.
- `name_core` = `name_norm` minus legal forms and stopwords (the, and, of, et, de, du, des, la, le, les, l').
- `text/variants.py`: from train GT positive pairs, align tokens (greedy best Jaro-Winkler per token) and count substitutions (a→b). Pairs with count ≥ 5 and precision ≥ 0.9 across GT are auto-added as extra canonicalization rules. Mined rules are fitted **within folds** when evaluating (fit on train folds only) and on full train for test.

**DoD:** unit tests on ~30 hand-written examples covering US, India, and French-style strings; normalized parquet cached.

---

## 6. Stage S2 — Candidate generation (blocking)

Goal: **recall ≥ 0.995** on OOF with the smallest candidate set possible. Report recall, reduction ratio, avg/median/max candidates per S1, per blocker and per country.

### 6.1 Blockers (all run S1→S2/S3 **and** S2/S3→S1; union)
| ID | Method | Default K |
|---|---|---|
| B1 | char_wb 2–4-gram TF-IDF on `name_core`, cosine top-K (chunked sparse matmul) | 30 |
| B2 | char_wb TF-IDF on `name_core + " " + addr_norm` | 30 |
| B3 | word TF-IDF (IDF-weighted) on name + address tokens | 30 |
| B4 | dense retrieval, pretrained `intfloat/multilingual-e5-base` (MIT), FAISS IP, input `"query: {name} | {address}"` | 30 |
| B5 | dense retrieval with **fine-tuned bi-encoder** (§6.2) | 30 |
| B6 | exact keys: same `postcode_like` + name TF-IDF > 0.25; shared rare name token (top-5% IDF); same `house_no` + street token | cap 30 |

Partition retrieval by the country value only if S0 showed 100% same-country positives; otherwise retrieve globally and let `same_country` be a feature.

### 6.2 Fine-tuned bi-encoder (B5)
- Base: `intfloat/multilingual-e5-base` (MIT). Loss: MultipleNegativesRankingLoss with in-batch negatives + 1 mined hard negative per positive (from B1–B4). 3 epochs, lr 2e-5, batch 64, max_len 96.
- **OOF:** train 5 fold models; each fold model retrieves for its held-out S1. Test retrieval uses the mean of the 5 fold embeddings (normalized). No full retrain needed.

### 6.3 Union, cap, prefilter
1. Union all blockers; record per pair: which blockers hit, each blocker's score and rank in both directions.
2. Cap per S1 at `K_union` (default 100) by max normalized blocker score.
3. **Prefilter model:** a fast LightGBM (OOF) on blocker scores/ranks + 10 cheap string features. Keep top `K_final` per S1 (default 20) **and** any pair with prefilter prob ≥ 0.01. Choose `K_final` as the smallest value keeping OOF recall ≥ 0.99 of the union's recall.
4. The post-prefilter set is **the final candidate set**. Every downstream model scores exactly this set, and this set is written to `candidate_pairs.tsv`.

**DoD:** recall table in `experiments.csv` and `reports/STATUS.md`; missed positives dumped to `reports/errors/blocking_misses.tsv` with both records side by side.

---

## 7. Stage S3 — Pair scoring models

### 7.1 Feature model (LightGBM, level-1)
Features on the final candidate set (vectorized, `rapidfuzz` + numpy):

*Name* (on `name_norm`, `name_core`, and best-over-`alias_names`): exact, ratio, partial_ratio, token_sort, token_set, WRatio, Jaro-Winkler, Levenshtein normalized, token Jaccard, IDF-weighted token overlap (soft, with JW ≥ 0.9 as match), char TF-IDF cosine (2–4 and 3–5 grams), e5 cosine (pretrained + fine-tuned OOF), first/last token match, acronym match, containment (one inside the other), legal form agree/conflict/missing (3-state), metaphone token overlap (`jellyfish`), length ratio, digit presence mismatch.

*Address*: same string sims on `addr_norm`; digits Jaccard; `postcode_like` equal/conflict/missing (3-state); `house_no` equal/conflict/missing; street-token overlap excluding numbers; landmark flag either side; address empty either side; name tokens appearing in the other's address.

*Context*: blocker hit bits, blocker scores and ranks (both directions), rank of candidate within S1 by each key sim, gap to S1's best, **mutual best** (S1 is the candidate's best S1 and vice versa), # candidates for S1, # candidates for the S2/S3 record, **name commonness** (# S1 records with same `name_core`; # test/train records sharing the top name token) — this is the main defense against chain-store false merges, source flag (S2/S3), `same_country`, cross-field swap sims (name vs address, detects swapped fields).

Training: OOF with the §4 folds, `objective=binary`, early stopping on fold val logloss, `seed` bagging ×3. Save OOF probs, test probs (mean over fold models), and feature importance.

### 7.2 Cross-encoder (level-1)
- Bases to try (in order): `xlm-roberta-base` (MIT), `xlm-roberta-large` (MIT), `microsoft/mdeberta-v3-base` (MIT).
- Text: `"name: {name} ; address: {address} ; country: {country}"` for each side; pair as (A, B); max_len 160.
- Training data: all positives + hard negatives from the final candidate set (top-scored by the prefilter), neg:pos ≤ 6:1. BCE loss; also try focal loss (γ=2). lr 1e-5 (large) / 2e-5 (base), 3 epochs, warmup 6%, bf16, gradient checkpointing for large.
- Augmentation: random A/B swap; train-time noise injection on one side (drop legal suffix, abbreviate/expand via abbrev map, drop postcode, swap token order, 1-char typo) with p=0.3 — this improves robustness to unseen formats.
- **OOF:** 5 fold models; test = mean over folds and over both A/B orderings.

### 7.3 LLM reranker (level-1, optional, see ladder gate)
- Base: an Apache-2.0 / MIT instruct model ≤ 8B (candidates: `Qwen/Qwen2.5-7B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`; verify license + param count on the model card at build time; per-size licenses differ within some families).
- Formulation: sequence-classification head **or** yes/no next-token logit on a fixed prompt containing both records. LoRA r=16, α=32, dropout 0.05 on attention+MLP projections, lr 1e-4, 1–2 epochs, bf16 (QLoRA 4-bit if on 24 GB).
- Cost control: train on positives + hardest negatives only; score only pairs whose level-1 stack prob is in the uncertain band [0.02, 0.98] (others get a sentinel + band flag feature). **OOF via 2 folds.**

### 7.4 Calibration
Isotonic regression on OOF for each level-1 model (fit per fold on out-of-fold data to avoid optimism). Save calibrators.

---

## 8. Stage S4 — Stacking and decoding

### 8.1 Group features (level-2 inputs)
For each pair (S1, c), from level-1 calibrated probs of all candidates of the same S1:
- max / 2nd max / sum of probs for this S1; this pair's rank and gap to max;
- for the S2/S3 record c: its max prob over other S1s, and whether this S1 is its argmax (conflict signal);
- **cluster consistency:** max similarity (CE-free, cheap string sim) between c and the other high-prob (≥0.5) candidates of this S1 — multiple S2/S3 duplicates of the same business reinforce each other;
- entropy of this S1's prob distribution.

### 8.2 Level-2 stacker
LightGBM (shallow: num_leaves 15, strong regularization) on [level-1 calibrated probs + group features + top-20 level-1 features]. Trained OOF on the same folds; final calibrated with isotonic. Compare against a simple logistic blend; keep the better by OOF macro F0.5.

### 8.3 Conflict resolution (1-to-1)
If S0 showed each S2/S3 ID appears in at most one GT list: sort all (S1, c) pairs by prob descending, assign each c to the first S1 that claims it, drop its other pairs. Also try: resolve only when the prob gap between the top two claimants ≥ δ, else drop c from both (precision play). Keep whichever wins on OOF.

### 8.4 Expected-F0.5 decoding (core of the approach)
For each S1 with calibrated candidate probs p₁ ≥ p₂ ≥ … ≥ pₙ (after 8.3):
- Consider predicted sets Sₖ = top-k candidates, k = 0..n (n ≤ 20).
- Expected score of k=0 (empty): ∏(1 − pᵢ) under independence.
- Expected score of k≥1: Monte Carlo with 512 samples of Bernoulli(pᵢ) (fixed RNG seed), computing the exact metric per sample including the "GT empty → 0" case. Use an exact DP instead if implemented and verified equal on tests.
- Pick k maximizing expected score.
- Independence is an approximation: add a tunable `prob_power γ` (pᵢ^γ) and `empty_bonus β` (multiply empty score by β), grid-search γ ∈ [0.7, 1.5], β ∈ [0.9, 1.2] on OOF.
- Baseline decoders to compare against: (a) global threshold t; (b) threshold t + relative α·max + singleton gate t_single. Keep the decoder with the best OOF macro F0.5; the expected-F decoder must beat (b) to be kept.

**DoD:** OOF macro F0.5, singleton accuracy, non-singleton F0.5, per-country, and LOCO — for every decoder variant — logged.

---

## 9. Stage S5 — Test-time adaptation (optional, gated)

Pseudo-label the test set to adapt to France:
1. Run the full pipeline on test.
2. Pseudo-positives: pairs with final prob ≥ 0.98 that survive 1-to-1 and are mutual best. Pseudo-negatives: in-candidate pairs with prob ≤ 0.01 (sample 6:1).
3. Fine-tune the fold cross-encoders for 1 extra epoch on train + pseudo-labels (pseudo weight 0.5), re-score test, re-stack.

**Gate:** simulate first with LOCO (treat eval country as unlabeled test, pseudo-label it, re-evaluate). Keep only if LOCO improves by ≥ 0.005 and in-distribution OOF doesn't drop. Test data is used without labels; document this clearly.

---

## 10. Stage S6 — Final fit and submission

- No full retrain: test predictions for every learned model = mean of fold models (identical to what OOF measured). Decoder hyperparameters come from OOF.
- `submit/write.py`:
  - asserts: rows == #test S1, all S1 present (count per country printed, France must be > 0), no dup rows/IDs, only S2-/S3- IDs, all exist in test, matches ⊆ candidates, candidate file == scored set;
  - writes both TSVs to `output/`;
  - runs `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test` and fails loudly if not PASS.
- Also write a **conservative variant** (`output/alt/`) using the decoder with the best LOCO score, in case it differs from the best OOF decoder. The human decides which to upload.
- Print prediction stats on test: % empty predictions per country vs train singleton rate; avg matches per S1. Flag if France's empty rate deviates > 15 pts from the train average.

---

## 11. Experiment tracking

`experiments.csv` columns:
`exp_id, timestamp, git_hash, stage, description, config_hash, blocking_recall, avg_candidates, reduction_ratio, oof_f05, oof_f05_singleton, oof_f05_nonsingleton, oof_f05_US, oof_f05_India, loco_f05, decoder, public_lb, runtime_min, kept(Y/N), notes`

- `git commit` before each experiment; the row's `git_hash` must match.
- `public_lb` is filled by humans; never used for selection.
- `reports/STATUS.md`: current best config, its scores, last 5 experiments, open issues. Keep under 60 lines.

---

## 12. Experiment ladder (run in order)

**Keep rule:** keep a change only if OOF macro F0.5 improves by ≥ 0.002 **and** LOCO does not drop by > 0.003. Otherwise revert. Log every attempt, including failures.

| # | Experiment | Notes |
|---|---|---|
| E1 | S0–S1, blockers B1–B3+B6, prefilter, LightGBM, threshold decoder (b) | **Baseline.** Write submission, validator PASS. |
| E2 | + 1-to-1 conflict resolution | if S0 supports it |
| E3 | + expected-F0.5 decoder | compare vs (b) |
| E4 | + B4 pretrained dense blocker, e5 cosine features | measure recall delta |
| E5 | + mined variant rules (§5) | |
| E6 | + fine-tuned bi-encoder B5 | |
| E7 | + cross-encoder xlm-roberta-base, stacked | |
| E8 | CE: focal loss, noise augmentation, xlm-roberta-large, mdeberta | one at a time |
| E9 | + group features + level-2 stacker | |
| E10 | seed bagging (GBM ×3, CE ×2 seeds on best config) | |
| E11 | + LLM reranker on uncertain band | **gate:** only if error analysis shows > 30% of remaining OOF errors are semantic (DBA/trade names, transliteration) rather than blocking misses or chains |
| E12 | pseudo-labeling (§9) | gated by LOCO simulation |
| E13 | ablation pass: remove each component of final config one at a time | drop anything that doesn't earn its keep; report in STATUS.md |

After E13, write `configs/final.yaml`, run `python -m src.cli all --config configs/final.yaml` from a clean `artifacts/`, and confirm the regenerated OOF score and output files match the logged ones.

---

## 13. Error analysis outputs (after every kept experiment)

Write to `reports/errors/`:
- `fp_top.tsv`, `fn_top.tsv`: 300 highest-confidence false positives / false negatives, both records side by side, with top-10 feature values and level-1 scores.
- `singleton_fp.tsv`: singletons where we predicted matches.
- `bucket_summary.md`: automatic bucketing of errors — blocking miss, chain/same-name, postcode conflict, transliteration-like (low char sim, high metaphone sim), DBA/alias, missing address, 1-to-1 conflict loss.

---

## 14. Packaging

1. `requirements.txt`: `pip freeze` filtered to imported packages, pinned.
2. `README.md`: instance types, setup commands, data placement, `python -m src.cli all --config configs/final.yaml`, per-stage commands, expected runtime per stage, model IDs + licenses, where OOF/test artifacts land, how to run tests.
3. `tests/` pass: `pytest -q`.
4. Zip layout:
```
<team_name>_submission.zip
├── output/{matching_results.tsv, candidate_pairs.tsv}
├── code/business_entity_resolution/{src/, configs/, tests/, README.md, requirements.txt}
└── Documentation_template.md
```
Documentation is written by teammates from `experiments.csv`, `reports/`, and this plan; the agent must keep those files accurate enough to be the source of truth.

---

## 15. Pre-flight checklist (agent runs before handing off)

- [ ] `pytest -q` passes.
- [ ] Clean re-run from raw data reproduces logged OOF score (±0.0005) and identical output TSVs.
- [ ] Validator PASS on `output/` and `output/alt/`.
- [ ] No country literals, no ID-derived features (grep checks).
- [ ] `reports/models.md` lists every model with license and param count.
- [ ] `reports/STATUS.md` states final OOF, LOCO, per-country, singleton accuracy, blocking recall, reduction ratio.
