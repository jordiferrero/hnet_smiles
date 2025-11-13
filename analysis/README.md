# H-Net SMILES Tokenization Analysis

This directory contains a comprehensive analysis framework for studying dynamic tokenization of SMILES/PSMILES chemical strings using H-Net models.

## Overview

This analysis studies how H-Net's dynamic chunking learns to tokenize chemical strings based on:
1. Dataset nature (polymer vs molecular)
2. Concatenation strategy
3. Training data amount
4. Comparison with SmilesPE benchmark

## Directory Structure

```
analysis/
├── utils/                      # Reusable Python utilities
│   ├── __init__.py
│   ├── inference.py           # Model loading and tokenization inference
│   ├── statistics.py          # Token statistics computation
│   └── benchmark.py           # SmilesPE benchmark utilities
├── data/                      # Generated data
│   ├── hnet_results/         # H-Net tokenization results (.pkl)
│   ├── smilesPE_results/     # SmilesPE tokenization results (.pkl)
│   ├── statistics/           # Computed statistics (.json)
│   └── SPE_ChEMBL.txt        # SmilesPE vocabulary file
├── notebooks/                 # Jupyter notebooks for analysis
│   ├── 01_data_generation.ipynb
│   ├── 02_dataset_nature_analysis.ipynb
│   ├── 03_concatenation_effect.ipynb
│   ├── 04_training_amount_analysis.ipynb
│   └── 05_benchmark_comparison.ipynb
├── figures/                   # Generated visualizations (.png)
├── README.md                  # This file
└── analysis_goals.md          # Original analysis goals document
```

## Analyzed Models

### 6 Trained H-Net Models:

1. **PI1M_concat_1epoch**: PI1M with 10-PSMILES concatenation, 1 epoch (68M bytes)
   - Path: `checkpoints/run_large_20251113_181705`

2. **PI1M_noconcat_5epoch**: PI1M no concatenation, 5 epoch (240M bytes)
   - Path: `checkpoints/run_large_20251111_075600`

3. **PI1M_concat_5epoch**: PI1M with 10-PSMILES concatenation, 5 epoch (240M bytes)
   - Path: `checkpoints/run_large_20251111_181836`

4. **PI1M_concat_22epoch**: PI1M with 10-PSMILES concatenation, 22 epoch (1B bytes)
   - Path: `checkpoints/run_large_20251112_150502`

5. **MOSES_noconcat_5epoch**: MOSES no concatenation, 5 epoch (360M bytes)
   - Path: `checkpoints/run_large_20251113_074900`

6. **MOSES_concat_5epoch**: MOSES with 10-SMILES concatenation, 5 epoch (360M bytes)
   - Path: `checkpoints/run_large_20251112_071557`

### Benchmarks:

- **SmilesPE on PI1M dataset**: Industry-standard tokenizer baseline
- **SmilesPE on MOSES dataset**: Industry-standard tokenizer baseline

## Workflow

### Step 1: Data Generation

**Notebook:** `01_data_generation.ipynb`

**GPU Required:** Yes (for H-Net inference)

This notebook:
1. Loads each H-Net model checkpoint
2. Runs tokenization inference on the full dataset
3. Computes and saves statistics
4. Runs SmilesPE benchmark (no GPU needed)

**Output:**
- `data/hnet_results/*.pkl` - Tokenization results for each model
- `data/smilesPE_results/*.pkl` - SmilesPE tokenization results
- `data/statistics/*.json` - Pre-computed statistics

**Important:** Set `RUN_HNET_INFERENCE=True` only when GPU is available!

### Step 2: Dataset Nature Analysis

**Notebook:** `02_dataset_nature_analysis.ipynb`

**Research Question (a):** How does dataset nature (polymer vs molecular) affect tokenization?

**Comparisons:**
- PI1M vs MOSES (non-concatenated, 5 epoch)
- PI1M vs MOSES (concatenated, 5 epoch)

**Outputs:**
- Token distribution comparisons
- Top token visualizations
- Breakpoint character analysis
- Summary metrics (Jaccard similarity, KL divergence)

### Step 3: Concatenation Effect Analysis

**Notebook:** `03_concatenation_effect.ipynb`

**Research Questions:**
- **b_1:** What is the effect of concatenation on learned tokens?
- **b_2:** Does concatenation affect polymers differently than molecules?

**Comparisons:**
- PI1M: no concat vs concat (5 epoch)
- MOSES: no concat vs concat (5 epoch)
- Magnitude of change: PI1M vs MOSES

**Outputs:**
- Concatenation effect metrics
- Dataset-specific sensitivity analysis

### Step 4: Training Amount Analysis

**Notebook:** `04_training_amount_analysis.ipynb`

**Research Question (c):** How does training data amount affect tokenization?

**Comparisons:**
- PI1M concat: 1 epoch vs 5 epoch vs 22 epoch (clean comparison across all training amounts)

**Outputs:**
- Token stability analysis
- Convergence patterns
- Overfitting indicators

### Step 5: Benchmark Comparison

**Notebook:** `05_benchmark_comparison.ipynb`

**Research Question (d):** How does H-Net compare to SmilesPE?

**Comparisons:**
- All PI1M models vs SmilesPE
- All MOSES models vs SmilesPE

**Outputs:**
- Comprehensive comparison table
- Token length comparisons
- Top token differences

## Key Utilities

### inference.py

```python
from analysis.utils.inference import load_model, run_tokenization_inference

# Load model
model, info = load_model('checkpoints/run_large_20251111_075600')

# Run inference
results = run_tokenization_inference(
    model, info['dataset_csv'], info['dataset_type'], device='cuda'
)
```

### statistics.py

```python
from analysis.utils.statistics import compute_token_statistics

# Compute statistics
stats = compute_token_statistics(results)

# Get summary
summary = stats.get_summary()
top_tokens = stats.get_top_tokens(50)
```

### benchmark.py

```python
from analysis.utils.benchmark import SmilesPEBenchmark

# Initialize SmilesPE
spe = SmilesPEBenchmark('analysis/data/SPE_ChEMBL.txt')

# Tokenize dataset
results = spe.tokenize_dataset('datasets/PI1M/PI1M_v2.csv', 'PI1M')
```

## Visualization Style

All plots use:
- **Style:** `seaborn whitegrid`
- **Context:** `talk` (suitable for presentations)
- **Color palette:** `mako`
- **DPI:** 300 (high resolution for A0 posters)

## Running the Analysis

1. **Setup environment:**
   ```bash
   source /opt/pytorch/bin/activate
   pip install SmilesPE  # Already done
   ```

2. **Generate data (when GPU available):**
   ```bash
   cd /home/ec2-user/hnet_smiles/analysis/notebooks
   jupyter notebook 01_data_generation.ipynb
   # Set RUN_HNET_INFERENCE=True and run all cells
   ```

3. **Run analyses:**
   ```bash
   # Run notebooks 02-05 in order
   # These don't require GPU, just load pre-computed statistics
   ```

## Important Notes

### GPU Usage

- **H-Net inference requires GPU** - Only run when training is not active
- **SmilesPE benchmark does NOT require GPU** - Can run anytime
- **Analysis notebooks do NOT require GPU** - They load pre-computed statistics

### Data Storage

- Tokenization results are stored as compressed pickle files (`.pkl`)
- Statistics are stored as JSON for easy inspection
- Large datasets are processed incrementally to manage memory

### Customization

All notebooks can be customized:
- Adjust number of top tokens/breakpoints to display
- Change visualization colors/styles
- Add additional metrics or comparisons
- Modify figure sizes for different poster formats

## Next Steps

After running all notebooks:
1. Review generated figures in `analysis/figures/`
2. Check summary CSV files in `analysis/data/`
3. Compile key findings into final report
4. Select poster-ready figures for A0 presentation

## Dependencies

- torch (for model loading)
- pandas, numpy (data processing)
- matplotlib, seaborn (visualization)
- SmilesPE (benchmark tokenizer)
- tqdm (progress bars)

All dependencies are already installed in the PyTorch environment.

