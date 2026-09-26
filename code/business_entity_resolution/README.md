# Business Entity Resolution Pipeline

This repository contains the complete machine learning pipeline for the Business Entity Resolution challenge.

It predicts which entity records (from sources 2 and 3) correspond to the same real-world business as the query entity (from source 1). The pipeline is robust to spelling variations, missing fields, legal suffixes, DBA names, and transliteration noise.

## 1. Setup Instructions

### Environment Setup

It is recommended to use a virtual environment. The pipeline requires Python 3.8+.

```bash
cd code/business_entity_resolution
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

This will install all required dependencies (listed in `pyproject.toml`) and expose the `ber-cli` command.

### Data Placement

Place your `.tsv` dataset files in the corresponding directories (the pipeline will read from here based on `configs/base.yaml`):

```
code/business_entity_resolution/
└── dataset/
    ├── train/
    │   ├── train_source1.tsv
    │   ├── train_source2.tsv
    │   ├── train_source3.tsv
    │   └── train_ground_truth.tsv
    └── test/
        ├── test_source1.tsv
        ├── test_source2.tsv
        └── test_source3.tsv
```

## 2. Running the Pipeline

The pipeline is driven via the command-line interface.

### Exploratory Data Analysis (S0)
To run a basic EDA on the training data and verify the files are loaded correctly:

```bash
ber-cli s0 --config configs/base.yaml
```
This will output a report to `reports/eda.md`.

### Full Pipeline (Ladder)
*Note: The CLI currently scaffolds the `s0` and `all` commands. To run the full experiment ladder (E1-E16), you will need to wire the respective module calls inside `src/cli.py` as your data becomes available.*

```bash
ber-cli all --config configs/base.yaml
```

## 3. Configuration

All hyperparameters, file paths, model revisions, and backend settings are centrally managed in `configs/base.yaml`.
- **`model.biencoder.base`**: HuggingFace base model for the B5 cross-encoder.
- **`decode.mode`**: Decoder strategy (`threshold_relative`, exact DP, etc.)
- **`blocking.k_max`**: Hard cap on candidate pool size.

Text normalization abbreviation rules (generic linguistic knowledge, not dataset-specific) are stored in `configs/abbrev.yaml`.

## 4. Pipeline Architecture

The pipeline consists of the following stages:

1. **Text Normalization (`src/text/`)**: Extracts clean text, postcodes, digits, phonetic skeletons, and applies abbreviation rules.
2. **Blocking (`src/blocking/`)**: Retrieves candidates using TF-IDF (B1-B3), Dense FAISS (B4-B5), and exact-key/phonetic anchors (B6-B8).
3. **Feature Engineering (`src/features/`)**: Computes string similarities (Jaccard, RapidFuzz, Monge-Elkan), context features, and group consistencies.
4. **Modeling (`src/models/`)**: Scores pairs using LightGBM, HuggingFace CrossEncoders, and an optional Rec2Rec same-entity model.
5. **Decoding (`src/decode/`)**: Resolves 1-to-1 conflicts and applies an exact Poisson-binomial DP decoder to maximize the expected F0.5 score.

## 5. Outputs

- Cached artifacts (models, features, fold splits) are saved to `artifacts/`.
- Final submission TSVs (`matching_results.tsv` and `candidate_pairs.tsv`) will be generated in the `output/` directory.
- You can validate the final submission files using the standalone script:
  ```bash
  python utils/validate_submission.py output/
  ```
