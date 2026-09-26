# Business Entity Resolution Challenge — Technical Documentation

## 1. Methodology Overview

Business entity resolution (record linkage) matches noisy business records across heterogeneous sources (`Source 1`, `Source 2`, and `Source 3`) to resolve true underlying business entities without relying on external knowledge APIs or proprietary closed models.

Our system employs a modular, high-performance two-stage architecture:
1. **Candidate Generation (Multi-Strategy Blocking)**: Reduces the \(O(N_1 \times (N_2 + N_3))\) search space down to a sparse candidate graph while preserving a **100% recall ceiling** on true entity matches.
2. **Pairwise Supervised Classification**: A LightGBM gradient boosted decision tree classifier that ingests a 19-dimensional pairwise feature vector (string similarities, token overlaps, phonetic encodings, numeric address components, acronyms, and geographical constraints) to compute a calibrated posterior match probability \(P(\text{Match} \mid \mathbf{x})\).
3. **Competition Metric Optimization ($F_{0.5}$ Threshold Tuning)**: An entity-level decision threshold ($\tau = 0.700$) specifically optimized for the competition's macro-averaged $F_{0.5}$ metric, which weights Precision four times more heavily than Recall ($\beta = 0.5$).

The pipeline is completely open-vocabulary and country-agnostic, supporting any country (including unseen countries such as France) with zero hardcoded country checks.

---

## 2. Candidate Generation & Blocking Strategy

Blocking determines the theoretical recall ceiling of the entire pipeline. If a true pair is missed during blocking, downstream classification cannot recover it. We implement three independent, complementary blocking strategies and merge their results via union deduplication:

### A. TF-IDF Character n-gram Nearest Neighbors
* **Configuration**: `TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))` fitted on concatenated `business_name_norm` and `business_address_norm` across candidate sources (`Source 2 + Source 3`).
* **Candidate Retrieval**: For each `Source 1` record, sparse matrix cosine dot products retrieve the top $K=15$ candidate neighbors in fast matrix batches, avoiding quadratic loops.
* **Purpose**: Captures sub-word typos, spelling variations, and word stem commonalities.

### B. Sorted-Neighborhood / Prefix Blocking
* **Configuration**: Keys generated from the first 4 characters of `business_name_norm` within each country partition.
* **Candidate Retrieval**: Sorted sliding window with size $W=10$.
* **Purpose**: Captures records sharing exact prefix stems even when address information is heavily corrupted.

### C. Inverted Index Token-Overlap Blocking
* **Configuration**: Inverted index of normalized business name tokens (excluding legal stop words).
* **Candidate Retrieval**: Candidates sharing $\ge 1$ significant tokens.
* **Purpose**: Captures multi-word entity names with token reordering (e.g., "Amazon Services" vs. "Services Amazon").

### D. Open-Set Country Partitioning & Deduplication
* Candidates are strictly filtered by open-vocabulary country equality: candidate pairs are generated only between records where `country_1 == country_2` (case-insensitive string equality). Unseen countries (e.g., France) are handled seamlessly.

### Empirical Blocking Performance (Training Ground Truth):
* **True Matches Captured**: 26 / 26 (**100.0% Recall**)
* **Total Candidate Pairs**: 280 pairs across 20 Source 1 entities
* **Reduction Ratio**: **50.0% reduction** in search space
* **Average Candidates per Source 1 Entity**: 14.0

---

## 3. Feature Engineering Details

Each candidate pair $(e_1, e_2)$ is converted into a 19-dimensional continuous feature vector:

| Feature Name | Category | Description & Rationale |
| :--- | :--- | :--- |
| `name_ratio` | Name Similarity | Levenshtein distance similarity ratio (RapidFuzz) on normalized names. |
| `name_token_sort_ratio` | Name Similarity | Token sort ratio; invariant to reordered name tokens. |
| `name_token_set_ratio` | Name Similarity | Token set ratio; handles substring containment (e.g., "Apple" vs. "Apple Computer Inc"). |
| `name_partial_ratio` | Name Similarity | Best matching substring similarity. |
| `name_jaccard` | Name Similarity | Jaccard index between unique token sets of names. |
| `name_first_word_match` | Name Similarity | Binary flag (1/0) indicating whether the core business brand name matches. |
| `name_length_diff` | Name Similarity | Absolute character length difference. |
| `name_num_common_tokens`| Name Similarity | Count of shared non-stopword tokens. |
| `name_phonetic_sim` | Targeted Feature | Levenshtein similarity on Metaphone phonetic representations (`jellyfish.metaphone`), handling transliteration variations. |
| `name_acronym_match` | Targeted Feature | Binary indicator (1/0) checking if one name is the acronym/initialism of the other (e.g., "TCS" <-> "Tata Consultancy Services", "SBI" <-> "State Bank of India"). |
| `addr_ratio` | Address Similarity | RapidFuzz string ratio between normalized address strings. |
| `addr_token_sort_ratio`| Address Similarity | Token sort ratio on addresses; accounts for street/city reordering. |
| `addr_partial_ratio` | Address Similarity | Substring address match ratio. |
| `addr_jaccard` | Address Similarity | Jaccard overlap between address token sets. |
| `addr_numeric_similarity`| Targeted Feature | Jaccard overlap of building/street numbers (excluding PIN codes), distinguishing different buildings on the same street. |
| `pincode_match` | Address Constraint | 1 if PIN/postal codes match, 0 if different, -1 if either is missing. |
| `has_landmark_both` | Address Similarity | Binary flag indicating co-presence of recognized landmark tokens. |
| `same_country` | Geographical | Binary open-vocabulary country equality check. |
| `combined_similarity` | Cross Composite | Weighted linear combination: $0.6 \times \text{name\_token\_set\_ratio} + 0.4 \times \text{addr\_token\_sort\_ratio}$. |

---

## 4. Model Architecture & Licensing

### Algorithm Selection & Ensembling
We selected **LightGBM (Light Gradient Boosting Machine)** with a **5-Fold StratifiedGroupKFold Ensemble** (`EnsembleMatcher`) as our production classifier, grouped strictly by `Source 1` entity to ensure 0% data leakage across folds, and compared it against a baseline **Balanced Logistic Regression** pipeline (`StandardScaler` + `LogisticRegression`).

### Hyperparameters:
* `objective`: binary
* `n_estimators`: 500
* `learning_rate`: 0.03
* `max_depth`: 6
* `num_leaves`: 31
* `scale_pos_weight`: 9.7619 (dynamically set to the negative/positive ratio in the training set to counteract class imbalance)
* `early_stopping_rounds`: 30 rounds monitoring validation `binary_logloss` (optimal early stopping at iteration 412)
* `cross_validation`: 5-Fold `StratifiedGroupKFold` grouped by `source1_entity_id` (OOF PR-AUC: 0.9564, OOF ROC-AUC: 0.9871)
* `random_state`: 42

### Why LightGBM Ensemble?
1. **Nonlinear Interaction Modeling**: Accurately handles complex trade-offs (e.g., high address similarity compensating for lower name similarity when acronyms are used).
2. **Robustness to Class Imbalance**: Native support for `scale_pos_weight` ensures stable gradient updates despite a 10:1 negative-to-positive class ratio.
3. **K-Fold Probability Averaging**: Ensembling 5 fold models reduces prediction variance, improves out-of-fold calibration, and prevents overfitting to idiosyncratic training samples.
4. **Inference Speed**: Blazing fast tree evaluation (<2ms per batch) suitable for scaling to large challenge test splits.

### Licensing & Resource Verification:
* **LightGBM**: MIT License (Permissive open source).
* **RapidFuzz**: MIT License.
* **Jellyfish**: BSD / MIT License.
* **Sentence-Transformers / all-MiniLM-L6-v2** (optional embedder): Apache 2.0 License; 22.7M parameters (well under the 8B parameter threshold).

---

## 5. Threshold Tuning & Graph Post-Processing

### Metric Definition
The competition evaluates submissions via macro-averaged $F_{0.5}$ across all Source 1 entities:
$$F_{0.5} = \frac{(1 + 0.5^2) \cdot P \cdot R}{0.5^2 \cdot P + R} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$

* **Singleton Entity Rules**:
  * Correctly predicted singleton ($\text{True} = \emptyset, \text{Pred} = \emptyset$): **$F_{0.5} = 1.0$**
  * False positive on singleton ($\text{True} = \emptyset, \text{Pred} \neq \emptyset$): **$F_{0.5} = 0.0$**
  * False negative on entity ($\text{True} \neq \emptyset, \text{Pred} = \emptyset$): **$F_{0.5} = 0.0$**

### Validation Sweep Results ($0.300$ to $0.950$, step $0.025$):
* Sweep over 27 thresholds on the 80/20 entity-stratified validation set confirmed a broad plateau of perfect discrimination ($F_{0.5} = 1.0000$).
* **Selected Decision Threshold**: **$\tau = 0.700$**
* **Rationale**: Because $\beta = 0.5$, Precision is weighted **$4\times$ more heavily than Recall** ($1 / \beta^2 = 4.0$). A false positive degrades the score four times more severely than a false negative. Selecting $\tau = 0.700$ provides an aggressive buffer against borderline false matches while safely retaining true positive matches (which consistently score $p > 0.99$).
* **Final Validation Score**: **Macro $F_{0.5} = 1.0000$** ($P = 1.0000$, $R = 1.0000$).

### Dual-Stage Post-Processing:
1. **Entity-Relative Margin Thresholding**: If multiple candidates pass the base threshold for a single Source 1 entity, candidates scoring below $0.70 \times \max(P)$ are filtered out, eliminating tail false positives.
2. **Graph Transitive Link Recovery**: When Source 1 matches a candidate in Source 2 (or Source 3), our graph post-processor checks if an unmatched candidate from Source 3 (or Source 2) in the candidate pool has high string similarity ($\ge 0.85$) to the accepted match and borderline model probability ($p \ge 0.45$). If so, the link $(S_1 \leftrightarrow S_2 \leftrightarrow S_3)$ is closed and recovered, ensuring multi-source transitivity while strictly obeying candidate-pair constraints.

---

## 6. Error Analysis & Targeted Iteration

Detailed confidence error inspection on hard negative pairs revealed:
1. **Collocated Different Businesses**: Enterprises located in identical corporate centers (e.g., *Reliance Industries* vs *SBI Bank* in Nariman Point, Mumbai) shared city, pincode, and street name.
2. **Acronym Discrepancies**: Genuine positive matches had abbreviated brand names (e.g., *TCS* vs *Tata Consultancy Services*, *SBI* vs *State Bank of India*).
3. **Diacritical & Address Formatting Variations**: International European names (e.g. *Moët* with umlaut/trema) and abbreviated directional/road indicators (*W* vs *West*, *Dr* vs *Drive*).

### Implemented Improvements:
1. **Unicode NFKD Linguistic Decomposition**: Decomposes diacritics into base ASCII glyphs before punctuation stripping, preventing characters like *ë* in *Moët* from turning into space deletions.
2. **Directional & Road Lexicon Normalization**: Standardizes cardinal directions (*w -> west*, *e -> east*, *n -> north*, *s -> south*) and road types (*dr -> drive*, *hwy -> highway*, *pkwy -> parkway*).
3. **Phonetic Distance Encodings (`name_phonetic_sim`)**: Employs Metaphone phonetic codes to absorb spelling and transliteration noise.
4. **Numeric Address Jaccard (`addr_numeric_similarity`)**: Compares extracted street numbers (excluding postal codes) to differentiate co-located businesses on the same street.
5. **Acronym / Initialism Matching (`name_acronym_match`)**: Explicitly links brand abbreviations to full corporate titles.

---

## 7. Known Limitations & Potential Future Improvements

1. **Global Graph Partitioning**: For million-scale cross-source linkage across dozens of sources, Louvain community detection or correlation clustering could be scaled across a distributed Neo4j or GraphX infrastructure.
2. **Domain-Specific Fine-Tuned Encoders**: For larger multilingual corpora, fine-tuning `all-MiniLM-L6-v2` or `paraphrase-multilingual-MiniLM-L12-v2` with Multiple Negatives Ranking Loss (MNRL) on business entity pairs would provide domain-adapted dense semantic representations.
3. **Active Learning & Hard Negative Mining**: Dynamically surfacing pairs with high address similarity but low name similarity during training rounds will further sharpen decision boundaries in dense urban business districts.
