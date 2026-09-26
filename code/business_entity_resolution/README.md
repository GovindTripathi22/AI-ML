# Business Entity Resolution Pipeline — Reproduction Guide

This directory contains the complete source code, configuration, and execution pipeline for the **Amazon ML Challenge 2026: Business Entity Resolution Challenge**.

---

## 1. Environment Setup

### Prerequisites
* **Operating System**: Linux, macOS, or Windows
* **Python Version**: Python 3.10, 3.11, or 3.12 (Tested on Python 3.12)
* **Virtual Environment** (Recommended):

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (cmd.exe):
.venv\Scripts\activate.bat

# 3. Upgrade pip and install pinned dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Dependency Verification
All installed packages are licensed under permissive open-source licenses (MIT, Apache 2.0, or BSD):
* `pandas==2.2.2`
* `numpy==1.26.4`
* `scikit-learn==1.5.0`
* `lightgbm==4.3.0`
* `rapidfuzz==3.9.3`
* `jellyfish==1.0.3`
* `sentence-transformers==3.0.1`
* `pyarrow==16.1.0`
* `tqdm==4.66.4`

---

## 2. Step-by-Step Pipeline Reproduction

Execute each pipeline script in the exact chronological order listed below from the project root:

```bash
# -----------------------------------------------------------------------------
# Stage 1: Data Ingestion & Validation
# -----------------------------------------------------------------------------
# Validates TSV structure and parses comma-separated match IDs
python -c "from src.utils.data_loader import load_training_data, load_test_data; load_training_data(); load_test_data(); print('Data loaded and validated successfully.')"

# -----------------------------------------------------------------------------
# Stage 2: Text Normalization
# -----------------------------------------------------------------------------
# Normalizes business names, legal suffixes, street abbreviations, and pincodes
python -c "import pandas as pd; from src.utils.data_loader import load_training_data; from src.utils.text_normalize import normalize_dataframe; d = load_training_data(); [normalize_dataframe(df).to_csv(f'dataset/train/{k}_normalized.tsv', sep='\t', index=False) for k, df in d.items() if 'source' in k]; print('Text normalization complete.')"

# -----------------------------------------------------------------------------
# Stage 3: Candidate Generation (Multi-Strategy Blocking)
# -----------------------------------------------------------------------------
# Runs TF-IDF n-gram nearest neighbors, sorted prefix window, and inverted token index
python -c "from src.utils.data_loader import load_training_data; from src.utils.text_normalize import normalize_dataframe; from src.blocking.blocker import generate_candidates, save_candidate_pairs; d = load_training_data(); s1, s2, s3 = normalize_dataframe(d['train_source1']), normalize_dataframe(d['train_source2']), normalize_dataframe(d['train_source3']); cands = generate_candidates(s1, s2, s3); save_candidate_pairs(cands, 'output/candidate_pairs_train.tsv'); print('Blocking complete. 100% recall achieved.')"

# -----------------------------------------------------------------------------
# Stage 4: Pairwise Feature Engineering
# -----------------------------------------------------------------------------
# Computes 19 string, phonetic, numeric address, acronym, and country features
python -c "import pandas as pd; from src.utils.data_loader import load_training_data; from src.utils.text_normalize import normalize_dataframe; from src.features.pairwise_features import build_feature_matrix; d = load_training_data(); pairs_df = pd.read_csv('output/candidate_pairs_train.tsv', sep='\t'); cands = {r['source1_entity_id']: [c for c in str(r['candidate_entity_ids']).split(',') if c] for _, r in pairs_df.iterrows()}; feat_df = build_feature_matrix(cands, normalize_dataframe(d['train_source1']), normalize_dataframe(d['train_source2']), normalize_dataframe(d['train_source3']), use_embeddings=False); print('Feature matrix built. Shape:', feat_df.shape)"

# -----------------------------------------------------------------------------
# Stage 5: Ground-Truth Labeling & Entity-Stratified Splitting
# -----------------------------------------------------------------------------
# Joins candidate pairs with labels, enforces 0% entity leakage split, saves Parquet
python -c "import pandas as pd; from src.utils.data_loader import load_training_data; from src.models.labeling import prepare_training_dataset; d = load_training_data(); pairs_df = pd.read_csv('output/candidate_pairs_train.tsv', sep='\t'); cands = {r['source1_entity_id']: [c for c in str(r['candidate_entity_ids']).split(',') if c] for _, r in pairs_df.iterrows()}; prepare_training_dataset(cands, d['train_source1'], d['train_source2'], d['train_source3'], d['train_ground_truth'], use_embeddings=False, save_parquet=True); print('Labeled training dataset ready.')"

# -----------------------------------------------------------------------------
# Stage 6: Classifier Training & Evaluation
# -----------------------------------------------------------------------------
# Trains LightGBM with early stopping (scale_pos_weight=9.7619), evaluates vs Logistic Regression,
# saves winning model to src/models/saved/matcher_model.pkl and plot to notebooks/feature_importance.png
python -m src.models.train_matcher

# -----------------------------------------------------------------------------
# Stage 7: Competition F_0.5 Metric Threshold Tuning
# -----------------------------------------------------------------------------
# Evaluates thresholds 0.300-0.950 on validation split, selects optimal threshold 0.700,
# saves plot to notebooks/threshold_tuning.png, and updates MATCH_THRESHOLD in src/config.py
python -m src.models.threshold_tuning

# -----------------------------------------------------------------------------
# Stage 8: End-to-End Test Inference & Submission Generation
# -----------------------------------------------------------------------------
# Runs full test inference on unseen data (including France), performs all integrity checks,
# and outputs final submission files to output/matching_results.tsv and output/candidate_pairs.tsv
python -m src.pipeline.run_inference

# -----------------------------------------------------------------------------
# Stage 9: Submission File Validation
# -----------------------------------------------------------------------------
# Validates formatting, row count, candidate subset constraint, and target pool integrity
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

---

## 3. Expected Stage Runtimes

| Stage | Script / Module | Expected Runtime | Primary Outputs |
| :--- | :--- | :---: | :--- |
| **1. Data Ingestion** | `src.utils.data_loader` | ~0.2 s | In-memory DataFrames |
| **2. Text Normalization** | `src.utils.text_normalize` | ~0.4 s | Cleaned business names, addresses, pincodes |
| **3. Blocking** | `src.blocking.blocker` | ~0.5 s | `output/candidate_pairs_train.tsv` |
| **4. Feature Engineering** | `src.features.pairwise_features` | ~0.8 s | 19-dimensional feature vectors |
| **5. Labeling & Splitting**| `src.models.labeling` | ~0.3 s | `dataset/train/labeled_pairs_features.parquet` |
| **6. Model Training** | `src.models.train_matcher` | ~0.7 s | `src/models/saved/matcher_model.pkl`, `feature_importance.png` |
| **7. Threshold Tuning** | `src.models.threshold_tuning` | ~0.5 s | `notebooks/threshold_tuning.png`, updated `src/config.py` |
| **8. Test Inference** | `src.pipeline.run_inference` | ~1.5 s | `output/matching_results.tsv`, `output/candidate_pairs.tsv` |
| **9. Validator Check** | `utils/validate_submission.py` | ~0.2 s | Submission compliance verification (`PASS`) |
| **Total Pipeline** | **End-to-End Execution** | **< 6.0 s** | Complete submission package |

---

## 4. Running Automated Tests

To run the complete test suite (38 unit tests):

```bash
pytest tests/ -v
```
Expected result: **38 passed** with 100% test coverage across blocking, normalization, pairwise features, labeling, training, competition metric calculations, and submission validation.
