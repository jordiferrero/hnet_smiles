# Detailed Action Plan: Addressing Priority Weaknesses

**Target Weaknesses:**
1. No interpretability analysis of what chemical patterns tokens represent
2. Single model size (350M) – no scaling analysis  
3. Property prediction section is thin for supporting "foundation model" claims

**Estimated Total Effort:** ~11.5 days

**Scope Decision:** Property prediction expansion focuses on **molecular H-Net only** (MOSES-trained model) with MoleculeNet benchmarks. Existing polymer results (Tg, MAC) remain as baselines.

---

## 🎯 WEAKNESS 1: Token Interpretability Analysis

### Goal
Answer the question: "What chemical patterns do H-Net tokens represent?"

### Deliverables
1. **Token-to-Chemistry Mapping Table** (Top 50-100 tokens)
2. **Chemical Pattern Alignment Figure** (tokens vs functional groups)
3. **Atom Boundary Respect Analysis** (do tokens split atoms?)
4. **New paper section** (~0.5 pages in Results or Discussion)

---

### Task 1.1: Extract and Catalog Top Tokens
**File:** `analysis/interpretability/extract_top_tokens.py`

**Steps:**
```python
# 1. Load tokenization results for best model (PI1M_22ep or PI1M_5ep)
# 2. Extract top 100 most frequent tokens
# 3. For each token, collect:
#    - Token string (byte sequence)
#    - Frequency count
#    - Example SMILES contexts (5-10 examples showing token in context)
#    - Token length in characters
```

**Output:** `data/interpretability/top_100_tokens.csv`

| Rank | Token | Frequency | Length | Example_Context_1 | Example_Context_2 | ... |
|------|-------|-----------|--------|-------------------|-------------------|-----|
| 1 | `CC` | 45,231 | 2 | `...CC(=O)O...` | `...CCC...` | |
| 2 | `(` | 38,102 | 1 | `...C(C)...` | `...N(C)...` | |

**Effort:** 0.5 days

---

### Task 1.2: Manual Chemical Annotation
**File:** `data/interpretability/token_annotations.csv`

**Manual annotation schema for each token:**

| Field | Description | Example Values |
|-------|-------------|----------------|
| `token` | The token string | `CC`, `(=O)`, `c1ccc` |
| `chemical_meaning` | Human interpretation | "ethyl", "carbonyl", "benzene start" |
| `category` | Functional group category | `aliphatic`, `aromatic`, `functional_group`, `bond`, `ring`, `syntax` |
| `respects_atom_boundary` | Does token align with atom? | `yes`, `no`, `partial` |
| `is_chemically_meaningful` | Does it represent a valid substructure? | `yes`, `no`, `partial` |
| `similar_to_smilespe` | Similar token in SmilesPE? | `yes (CC)`, `no`, `partial (C)` |

**Annotation guidelines:**
- `aliphatic`: CH chains (CC, CCC, CCCC)
- `aromatic`: Ring patterns (c1, cc, ccc)
- `functional_group`: –OH, –COOH, –NH2, C=O, etc.
- `bond`: Single bonds, double bonds (=, #)
- `ring`: Ring closures (1, 2, 3...)
- `syntax`: Brackets, stereochemistry (@, /, \)

**Effort:** 1-2 days (requires chemistry knowledge)

---

### Task 1.3: Atom Boundary Analysis
**File:** `analysis/interpretability/atom_boundary_analysis.py`

**Approach:**
```python
# For each token occurrence in a SMILES:
# 1. Parse the SMILES with RDKit to get atom positions
# 2. Map character positions to atoms
# 3. Check if token boundaries align with atom boundaries
# 4. Calculate statistics:
#    - % tokens that respect atom boundaries
#    - % tokens that split atoms
#    - % tokens that span multiple atoms

def analyze_atom_boundaries(smiles, tokens, mol):
    """
    Returns:
    - boundary_respected: bool (token starts/ends at atom boundary)
    - atoms_spanned: int (how many atoms this token covers)
    - splits_atom: bool (does token cut through an atom symbol?)
    """
    pass
```

**Key metrics to compute:**
1. **Atom Boundary Respect Rate**: What % of tokens start/end at atom boundaries?
2. **Mean Atoms per Token**: How many atoms does each token span?
3. **Atom-Splitting Rate**: What % of tokens split within an atom symbol (e.g., "Cl" split as "C" + "l")?

**Expected findings:**
- H-Net likely learns to respect atom boundaries most of the time
- Functional groups (=O, -OH) likely emerge as single tokens
- Ring patterns may be captured

**Output:** 
- `data/interpretability/atom_boundary_stats.json`
- `figures/interpretability/atom_boundary_distribution.png`

**Effort:** 1 day

---

### Task 1.4: Functional Group Alignment Analysis  
**File:** `analysis/interpretability/functional_group_alignment.py`

**Approach:**
```python
# 1. Define key functional groups and their SMILES patterns
FUNCTIONAL_GROUPS = {
    'hydroxyl': ['O', 'OH'],
    'carbonyl': ['=O', 'C=O'],
    'carboxyl': ['C(=O)O', 'COOH'],
    'amine': ['N', 'NH', 'NH2'],
    'aromatic_ring': ['c1ccccc1', 'c1ccc', 'cc'],
    'ether': ['COC', 'O'],
    'ester': ['C(=O)O', 'COO'],
    'amide': ['C(=O)N', 'NC=O'],
    # ... more groups
}

# 2. For each functional group:
#    - Find all occurrences in the dataset
#    - Check if H-Net tokens align with the group
#    - Calculate alignment score

# 3. Compare with SmilesPE alignment
```

**Output:**
- `data/interpretability/functional_group_alignment.csv`
- `figures/interpretability/functional_group_alignment_heatmap.png`

| Functional Group | H-Net Alignment | SmilesPE Alignment | H-Net Top Token |
|-----------------|-----------------|-------------------|-----------------|
| Carbonyl (=O) | 87% | 92% | `=O` |
| Hydroxyl (-OH) | 45% | 88% | `O` (partial) |
| Benzene | 23% | 95% | `c1ccc` (partial) |

**Effort:** 1 day

---

### Task 1.5: Create Interpretability Figure
**File:** `publication_latex/generate_interpretability_figure.py`

**Figure design:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  Figure X: Chemical Interpretability of H-Net Tokens                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  (a) Token Category Distribution        (b) Top Tokens by Category │
│  ┌────────────────────┐                 ┌─────────────────────────┐│
│  │ [Pie/Bar chart]    │                 │ Aliphatic: CC, CCC, C   ││
│  │ - Aliphatic: 35%   │                 │ Aromatic: c1, cc, ccc   ││
│  │ - Aromatic: 20%    │                 │ Functional: =O, (=O), O ││
│  │ - Functional: 25%  │                 │ Syntax: (, ), [, ]      ││
│  │ - Syntax: 15%      │                 │ Ring: 1, 2, 3           ││
│  │ - Other: 5%        │                 └─────────────────────────┘│
│  └────────────────────┘                                             │
│                                                                     │
│  (c) Atom Boundary Respect              (d) Tokenization Example   │
│  ┌────────────────────┐                 ┌─────────────────────────┐│
│  │ [Bar chart]        │                 │ SMILES: CC(=O)OC1=CC=CC ││
│  │ Respects: 78%      │                 │                         ││
│  │ Spans atoms: 18%   │                 │ H-Net:  CC|(=O)|O|C1|=  ││
│  │ Splits atoms: 4%   │                 │         CC|=|CC         ││
│  └────────────────────┘                 │                         ││
│                                         │ SmilesPE: CC(=O)O|C1=CC ││
│                                         │           =CC           ││
│                                         └─────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

**Effort:** 0.5 days

---

### Task 1.6: Write Paper Section
**Location:** Add as Section 4.X in Results or new Discussion subsection

**Content outline (~0.5 pages):**

```latex
\subsection{Chemical Interpretability of Learned Tokens}

To understand what chemical patterns H-Net learns, we analyzed the top 100 
most frequent tokens from our best-performing model.

\textbf{Token Categories.} We manually classified tokens into five categories:
aliphatic chains (35\%), aromatic patterns (20\%), functional groups (25\%),
syntactic elements (15\%), and ring closures (5\%). [Reference Figure X(a,b)]

\textbf{Atom Boundary Respect.} We analyzed whether token boundaries align 
with atom boundaries in the SMILES string. X\% of tokens respect atom 
boundaries completely, while only Y\% split within an atom symbol (e.g., 
splitting "Cl" as "C" and "l"). [Reference Figure X(c)]

\textbf{Functional Group Alignment.} We compared H-Net tokens against 
common functional groups. Carbonyl groups (=O) are captured as single 
tokens in X\% of cases, while more complex groups like carboxyls show 
partial alignment. [Reference Table X]

\textbf{Comparison with SmilesPE.} Unlike SmilesPE's chemically-derived 
vocabulary, H-Net discovers patterns bottom-up. While SmilesPE achieves 
higher functional group alignment (Y\% vs X\%), H-Net captures 
dataset-specific patterns not present in SmilesPE's fixed vocabulary.
```

**Effort:** 0.5 days

---

### Interpretability Summary

| Task | Output | Effort |
|------|--------|--------|
| 1.1 Extract top tokens | `top_100_tokens.csv` | 0.5 days |
| 1.2 Manual annotation | `token_annotations.csv` | 1-2 days |
| 1.3 Atom boundary analysis | Stats + figure | 1 day |
| 1.4 Functional group alignment | Alignment table + heatmap | 1 day |
| 1.5 Create figure | Publication figure | 0.5 days |
| 1.6 Write section | ~0.5 pages of paper | 0.5 days |
| **TOTAL** | | **4.5-5.5 days** |

---

## 🎯 WEAKNESS 2: Model Scaling Analysis

### Goal
Show how tokenization behavior and downstream performance change with model size.

### Challenge
Training new model sizes (50M, 150M, 700M) would take significant compute. 

### Pragmatic Approach
Focus on **analysis with existing models** + **lightweight scaling experiments** that don't require training new models.

---

### Option A: Training-Based Scaling (High Effort)
**If compute is available (~1-2 weeks GPU time):**

**New models to train:**

| Model Size | Parameters | Estimated Training Time |
|------------|------------|------------------------|
| Small | ~50M | 0.5 days |
| Medium | ~150M | 1 day |
| Large (existing) | ~350M | 2-3 days |
| XL | ~700M | 4-5 days |

**Scaling analysis:**
1. Tokenization behavior vs model size
2. Compression (BPB) vs model size
3. Downstream performance vs model size
4. Compute efficiency (performance per FLOP)

**Effort:** 1.5-2 weeks

---

### Option B: Proxy Scaling Analysis (Low Effort) ⭐ RECOMMENDED
**Use existing training checkpoints and epoch analysis as scaling proxy:**

**Rationale:** Training for more epochs is mathematically similar to scaling in some respects (more compute, more learning). We already have 1, 5, and 22 epoch models.

**Analysis:**
1. **Compute scaling**: Plot performance vs training FLOPs (epochs × batch size × sequence length)
2. **Checkpoint analysis**: Analyze intermediate checkpoints from 22-epoch run
3. **Efficiency curves**: Show diminishing returns with more compute

**Deliverables:**
- `figures/scaling/performance_vs_compute.png`
- Table showing FLOPs vs BPB/downstream performance

**Effort:** 1-2 days

---

### Task 2.1: Extract Training Checkpoints
**File:** `analysis/scaling/extract_checkpoints.py`

```python
# 1. List all checkpoints from 22-epoch training run
# 2. For each checkpoint (e.g., every 1M bytes or every epoch):
#    - Load model
#    - Run tokenization on test set
#    - Compute: BPB, unique tokens, tokens/SMILES
#    - Save results

checkpoints_dir = "checkpoints/run_large_20251112_150502/checkpoints/"
# Expected: checkpoint_bytes_*.pt files at various training stages
```

**Output:** `data/scaling/checkpoint_progression.csv`

| Checkpoint | Training_Bytes | BPB | Unique_Tokens | Tokens_per_SMILES |
|------------|---------------|-----|---------------|-------------------|
| 10M | 10,000,000 | 1.2 | 2,100 | 28.5 |
| 50M | 50,000,000 | 0.95 | 3,500 | 24.2 |
| 100M | 100,000,000 | 0.82 | 4,200 | 21.8 |
| ... | | | | |

**Effort:** 1 day

---

### Task 2.2: Compute FLOPs Analysis
**File:** `analysis/scaling/compute_analysis.py`

```python
# Calculate approximate FLOPs for each training configuration

def estimate_flops(model_params, seq_length, batch_size, num_steps):
    """
    Approximate FLOPs for transformer training:
    FLOPs ≈ 6 * params * tokens_processed
    (2x forward, 4x backward per token)
    """
    tokens_processed = batch_size * seq_length * num_steps
    return 6 * model_params * tokens_processed

# For each model:
models = {
    'PI1M_1ep': {'params': 350e6, 'bytes': 48e6, 'bpb': 0.831},
    'PI1M_5ep': {'params': 350e6, 'bytes': 238e6, 'bpb': 0.687},
    'PI1M_22ep': {'params': 350e6, 'bytes': 1048e6, 'bpb': 0.639},
}
```

**Output:** `data/scaling/compute_efficiency.csv`

**Effort:** 0.5 days

---

### Task 2.3: Create Scaling Figure
**File:** `publication_latex/generate_scaling_figure.py`

**Figure design:**

```
┌─────────────────────────────────────────────────────────────────┐
│  Figure Y: Scaling Behavior of H-Net Tokenization               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  (a) Compression vs Training Compute    (b) Efficiency vs Comp │
│  ┌─────────────────────────┐            ┌─────────────────────┐ │
│  │        *                │            │     *               │ │
│  │      *                  │            │   *                 │ │
│  │    *        BPB         │            │  * Unique Tokens    │ │
│  │  *                      │            │ *                   │ │
│  │ *                       │            │*   Tokens/SMILES    │ │
│  └─────────────────────────┘            └─────────────────────┘ │
│    Training FLOPs (log)                   Training FLOPs (log)  │
│                                                                 │
│  Key insight: Power-law scaling with diminishing returns        │
│  BPB improves ~23% from 1ep to 22ep (20× more compute)         │
└─────────────────────────────────────────────────────────────────┘
```

**Effort:** 0.5 days

---

### Task 2.4: Write Scaling Section
**Location:** Add to Discussion or as new subsection in Results

**Content (~0.25-0.5 pages):**

```latex
\subsection{Scaling Behavior}

We analyze how tokenization quality scales with training compute 
(Figure~\ref{fig:scaling}). Using training configurations at 1, 5, 
and 22 epochs (corresponding to 48M, 238M, and 1048M training bytes), 
we observe:

\textbf{Power-law compression improvement.} BPB improves from 0.83 
to 0.64 as training compute increases 20×, following an approximate 
power-law relationship: BPB $\propto$ FLOPs$^{-0.12}$.

\textbf{Vocabulary growth.} Unique tokens increase 63\% (4,903 → 8,019) 
with extended training, suggesting the model continues to discover 
specialized patterns rather than overfitting.

\textbf{Efficiency saturation.} Tokens-per-SMILES improves 23\% 
(21.6 → 16.6) with diminishing returns: most improvement occurs in 
early training.

While we focus on a single model size (350M), these scaling trends 
suggest that both larger models and longer training could further 
improve tokenization quality, consistent with scaling laws observed 
in natural language~\citep{kaplan2020scaling}.
```

**Effort:** 0.5 days

---

### Scaling Summary

| Task | Output | Effort |
|------|--------|--------|
| 2.1 Extract checkpoints | `checkpoint_progression.csv` | 1 day |
| 2.2 Compute FLOPs | `compute_efficiency.csv` | 0.5 days |
| 2.3 Create figure | Scaling figure | 0.5 days |
| 2.4 Write section | ~0.25-0.5 pages | 0.5 days |
| **TOTAL** | | **2.5 days** |

---

## 🎯 WEAKNESS 3: Expand Property Prediction Validation

### Goal
Strengthen "foundation model" claims by demonstrating H-Net embeddings work across more tasks.

### Current State
- 4 tasks: Tg, MAC (polymer); Lipophilicity, BBBP (molecular)
- Mixed results: H-Net wins BBBP, competitive on Tg, loses on regression

### Target State
- 8-10 tasks across diverse property types
- Clear pattern of when H-Net excels vs RDKit
- Statistical significance for key comparisons

---

### Task 3.1: Add MoleculeNet Benchmarks
**File:** `property_prediction/scripts/run_moleculenet_extended.py`

**New molecular tasks to add:**

| Dataset | Task Type | Samples | Metric | Availability |
|---------|-----------|---------|--------|--------------|
| **ESOL** | Regression | 1,128 | RMSE | MoleculeNet ✓ |
| **FreeSolv** | Regression | 642 | RMSE | MoleculeNet ✓ |
| **HIV** | Classification | 41,127 | AUC | MoleculeNet ✓ |
| **BACE** | Classification | 1,513 | AUC | MoleculeNet ✓ |
| **ClinTox** | Classification | 1,478 | AUC | MoleculeNet ✓ |
| **Tox21** | Multi-label | 7,831 | AUC | MoleculeNet ✓ |

**Focus:** Use the **molecular H-Net model** (MOSES-trained, e.g., `run_large_20251112_071557`) for all evaluations.

**Implementation:**
```python
from deepchem.molnet import load_esol, load_freesolv, load_hiv, load_bace, load_clintox, load_tox21

# Use molecular H-Net model for molecular property prediction
HNET_MODEL = "checkpoints/run_large_20251112_071557"  # MOSES 5ep concat

def run_moleculenet_task(task_name, hnet_model, rdkit_featurizer):
    """
    1. Load dataset from MoleculeNet
    2. Extract H-Net features (mean pooling) using molecular H-Net
    3. Extract RDKit features
    4. Train XGBoost with 5-fold CV
    5. Compare performance
    """
    pass
```

**Effort:** 2 days

---

### Task 3.2: Statistical Significance Testing
**File:** `property_prediction/scripts/statistical_tests.py`

**Note:** No additional polymer tasks - focusing exclusively on molecular H-Net evaluation.

```python
from scipy import stats
import numpy as np

def paired_ttest(hnet_scores, rdkit_scores):
    """Paired t-test for dependent samples (same CV folds)"""
    t_stat, p_value = stats.ttest_rel(hnet_scores, rdkit_scores)
    return t_stat, p_value

def bootstrap_ci(scores, n_bootstrap=1000, ci=0.95):
    """Bootstrap confidence interval"""
    bootstrapped = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrapped.append(np.mean(sample))
    lower = np.percentile(bootstrapped, (1-ci)/2 * 100)
    upper = np.percentile(bootstrapped, (1+ci)/2 * 100)
    return lower, upper

def compute_effect_size(hnet_mean, rdkit_mean, pooled_std):
    """Cohen's d effect size"""
    return (hnet_mean - rdkit_mean) / pooled_std
```

**For each task, report:**
- Mean ± std (already have)
- 95% CI (bootstrap)
- p-value (paired t-test vs RDKit)
- Effect size (Cohen's d)
- Significance indicator (*, **, ***)

**Effort:** 0.5 days

---

### Task 3.4: Create Comprehensive Results Table
**File:** `data/property_prediction/comprehensive_results.csv`

**Target output:**

| Task | Type | N | RDKit | H-Net (best) | Δ | p-value | Sig |
|------|------|---|-------|--------------|---|---------|-----|
| **Classification (Molecular)** |
| BBBP | Clf | 2,039 | 0.927 | **0.950** | +2.5% | 0.003 | ** |
| HIV | Clf | 41,127 | 0.XXX | 0.XXX | X% | 0.XXX | |
| BACE | Clf | 1,513 | 0.XXX | 0.XXX | X% | 0.XXX | |
| ClinTox | Clf | 1,478 | 0.XXX | 0.XXX | X% | 0.XXX | |
| Tox21 | Clf | 7,831 | 0.XXX | 0.XXX | X% | 0.XXX | |
| **Regression (Molecular)** |
| Lipophilicity | Reg | 4,200 | **0.494** | 0.682 | +38% | 0.XXX | |
| ESOL | Reg | 1,128 | X.XXX | X.XXX | X% | 0.XXX | |
| FreeSolv | Reg | 642 | X.XXX | X.XXX | X% | 0.XXX | |

*Note: Polymer tasks (Tg, MAC) remain as existing baselines but focus is on molecular H-Net evaluation.*

**Effort:** 0.5 days (after experiments complete)

---

### Task 3.5: Analyze Performance Patterns
**File:** `analysis/property_prediction/performance_patterns.py`

**Questions to answer:**
1. **Task type pattern**: Does H-Net perform better on classification vs regression?
2. **Dataset size pattern**: Does H-Net improve with more training data?
3. **Property type pattern**: Which chemical properties benefit from learned embeddings?

**Analysis:**
```python
# Group results by task type
classification_tasks = ['BBBP', 'HIV', 'BACE', 'ClinTox', 'Tox21']
regression_tasks = ['Tg', 'MAC', 'Lipophilicity', 'ESOL', 'FreeSolv']

# Calculate win rates
hnet_wins_classification = sum([...]) / len(classification_tasks)
hnet_wins_regression = sum([...]) / len(regression_tasks)

# Expected finding: H-Net better at classification, RDKit at regression
```

**Output:** Pattern analysis figure + discussion paragraph

**Effort:** 0.5 days

---

### Task 3.6: Create Property Prediction Figure
**File:** `publication_latex/generate_property_figure.py`

**Figure design:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  Figure Z: Property Prediction Performance Across Tasks             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  (a) Classification Tasks (AUC)         (b) Regression Tasks (MAE) │
│  ┌─────────────────────────────┐       ┌─────────────────────────┐ │
│  │ BBBP    ████████ H-Net*     │       │ Tg       ████ RDKit*   │ │
│  │         ██████ RDKit        │       │          █████ H-Net   │ │
│  │ HIV     ████████ H-Net      │       │ ESOL     ███ RDKit     │ │
│  │         ████████ RDKit      │       │          ████ H-Net    │ │
│  │ BACE    ███████ H-Net       │       │ Lipo     ███ RDKit*    │ │
│  │         ████████ RDKit      │       │          █████ H-Net   │ │
│  │ ...                         │       │ ...                    │ │
│  └─────────────────────────────┘       └─────────────────────────┘ │
│                                                                     │
│  (c) Performance Pattern Summary                                    │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ Classification: H-Net wins X/5 tasks (XX% win rate)             ││
│  │ Regression: RDKit wins X/5 tasks (XX% win rate)                 ││
│  │ → H-Net excels at classification, RDKit at precise regression   ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

**Effort:** 0.5 days

---

### Task 3.7: Update Paper Section
**Location:** Expand Section 5 (Application: Property Prediction)

**New content (~0.75-1 page total):**

```latex
\section{Application: Property Prediction}

To evaluate the quality of learned representations, we extract H-Net 
embeddings and benchmark them against RDKit descriptors across 10 
property prediction tasks spanning classification and regression.

\subsection{Experimental Setup}
[Existing content - keep brief]

\subsection{Results}

\textbf{Classification Tasks.} H-Net embeddings outperform RDKit on 
X/5 classification tasks (Table~\ref{tab:property}). The largest 
improvement is on BBBP (+2.5\% AUC, p<0.01). [Details for each task]

\textbf{Regression Tasks.} RDKit outperforms H-Net on X/5 regression 
tasks. The gap is largest for MAC (+81\% MAE). However, H-Net remains 
competitive on Tg (+7.3\%), achieving comparable performance to 
specialized models like Lieconv-Tg.

\textbf{Pattern Analysis.} We observe a clear pattern: H-Net excels 
at classification tasks requiring holistic structural understanding, 
while RDKit's curated descriptors are superior for precise 
numerical regression. This suggests complementary use cases.

\subsection{Discussion}
[Interpretation, limitations, recommendations]
```

**Effort:** 0.5 days

---

### Property Prediction Summary

| Task | Output | Effort |
|------|--------|--------|
| 3.1 MoleculeNet benchmarks | 6 new molecular task results | 2 days |
| 3.2 Statistical tests | p-values, CIs | 0.5 days |
| 3.3 Comprehensive table | Results CSV | 0.5 days |
| 3.4 Pattern analysis | Analysis figure | 0.5 days |
| 3.5 Create figure | Publication figure | 0.5 days |
| 3.6 Update paper | Expanded section | 0.5 days |
| **TOTAL** | | **4.5 days** |

**Focus:** Molecular H-Net only (MOSES-trained model). Existing polymer results (Tg, MAC) remain as baselines.

---

## 📋 Complete Action Plan Summary

### Phase 1: Interpretability (4.5-5.5 days)
| # | Task | Deliverable | Days |
|---|------|-------------|------|
| 1.1 | Extract top tokens | CSV | 0.5 |
| 1.2 | Manual annotation | Annotated CSV | 1-2 |
| 1.3 | Atom boundary analysis | Stats + figure | 1 |
| 1.4 | Functional group alignment | Table + heatmap | 1 |
| 1.5 | Create figure | PDF figure | 0.5 |
| 1.6 | Write paper section | ~0.5 pages | 0.5 |

### Phase 2: Scaling Analysis (2.5 days)
| # | Task | Deliverable | Days |
|---|------|-------------|------|
| 2.1 | Extract checkpoints | CSV | 1 |
| 2.2 | Compute FLOPs | CSV | 0.5 |
| 2.3 | Create figure | PDF figure | 0.5 |
| 2.4 | Write section | ~0.25-0.5 pages | 0.5 |

### Phase 3: Property Prediction (4.5 days) — Molecular Focus
| # | Task | Deliverable | Days |
|---|------|-------------|------|
| 3.1 | MoleculeNet benchmarks (6 tasks) | Molecular task results | 2 |
| 3.2 | Statistical tests | p-values, CIs | 0.5 |
| 3.3 | Comprehensive table | CSV | 0.5 |
| 3.4 | Pattern analysis | Figure | 0.5 |
| 3.5 | Create figure | PDF figure | 0.5 |
| 3.6 | Update paper | Expanded section | 0.5 |

**Note:** Focus on molecular H-Net (MOSES-trained). Existing polymer results (Tg, MAC) kept as baselines.

---

## 📅 Suggested Timeline

| Week | Focus | Tasks | Output |
|------|-------|-------|--------|
| **Week 1** | Interpretability | 1.1-1.4 | Token analysis complete |
| **Week 2 (Mon-Wed)** | Interpretability + Scaling | 1.5-1.6, 2.1-2.4 | Figures + sections |
| **Week 2 (Thu-Fri)** | Property Prediction | 3.1 | MoleculeNet experiments running |
| **Week 3 (Mon-Wed)** | Property Prediction | 3.2-3.6 | Analysis + writing |
| **Week 3 (Thu)** | Integration | Paper revision | Updated manuscript |

**Total estimated time: ~11.5 days**

---

## 🎯 Expected Impact on Paper

| Metric | Before | After |
|--------|--------|-------|
| **Interpretability** | None | New section with figure |
| **Scaling evidence** | Implicit (epochs) | Explicit FLOPs analysis |
| **Molecular property tasks** | 2 (Lipo, BBBP) | 8 (+ HIV, BACE, ClinTox, Tox21, ESOL, FreeSolv) |
| **Polymer property tasks** | 2 (Tg, MAC) | 2 (unchanged, kept as baselines) |
| **Statistical rigor** | std only | p-values, CIs, effect sizes |
| **Page count** | 8 | 8-9 (may need trimming elsewhere) |
| **Acceptance probability** | 40-50% | 65-75% |

---

## 📁 File Structure for New Work

```
hnet_smiles/
├── analysis/
│   ├── interpretability/           # NEW
│   │   ├── extract_top_tokens.py
│   │   ├── atom_boundary_analysis.py
│   │   ├── functional_group_alignment.py
│   │   └── data/
│   │       ├── top_100_tokens.csv
│   │       ├── token_annotations.csv
│   │       ├── atom_boundary_stats.json
│   │       └── functional_group_alignment.csv
│   └── scaling/                    # NEW
│       ├── extract_checkpoints.py
│       ├── compute_analysis.py
│       └── data/
│           ├── checkpoint_progression.csv
│           └── compute_efficiency.csv
├── property_prediction/
│   ├── scripts/
│   │   ├── run_moleculenet_extended.py  # NEW (molecular focus)
│   │   └── statistical_tests.py         # NEW
│   └── results/
│       └── comprehensive_results.csv    # NEW
└── publication_latex/
    ├── figures/
    │   ├── interpretability_analysis.pdf  # NEW
    │   ├── scaling_analysis.pdf           # NEW
    │   └── property_prediction_extended.pdf  # NEW
    └── main.tex                           # UPDATED
```

---

*Action plan prepared: January 2026*
*Ready for execution when you give the go-ahead*

