# H-Net as Chemical Featurizer: Property Prediction Study

This subproject evaluates trained H-Net models as chemical featurizers for property prediction tasks.

## Overview

We compare H-Net latent representations against:
- **RDKit descriptors**: Traditional chemical fingerprints and physicochemical properties
- **Lieconv-Tg**: SOTA model for polymer glass transition temperature (Tg)
- **SMI-TED**: SOTA transformer model for molecular property prediction

## Tasks

### Polymer Properties (PSMILES)
- **Tg (Glass Transition Temperature)**: Regression, MAE metric
- **MAC (Mass Attenuation Coefficient)**: Regression, MAE metric

### Molecule Properties (MoleculeNet)
- **BBBP (Blood-Brain Barrier Penetration)**: Classification, AUC metric - **H-Net wins (+2.5%)**
- **HIV (HIV replication inhibition)**: Classification, AUC metric - **H-Net wins (+3.7%)**
- **BACE (β-secretase 1 inhibitors)**: Classification, AUC metric
- **Lipophilicity**: Regression, MAE metric
- **ESOL (Aqueous Solubility)**: Regression, RMSE metric
- **FreeSolv (Hydration Free Energy)**: Regression, RMSE metric

## Key Results (Jan 2026)

| Task | RDKit | H-Net | Winner |
|------|-------|-------|--------|
| BBBP (AUC) | 0.927 | **0.950** | H-Net ✓ |
| HIV (AUC) | 0.760 | **0.788** | H-Net ✓ |
| BACE (AUC) | **0.897** | 0.867 | RDKit |
| Lipophilicity (MAE) | **0.494** | 0.682 | RDKit |
| ESOL (RMSE) | **0.660** | 0.910 | RDKit |

**Conclusion**: H-Net excels at classification tasks, RDKit wins on regression.

## H-Net Models Evaluated

### Polymer Models (5 variants)
| Checkpoint | Epochs | Concat | Architecture |
|------------|--------|--------|--------------|
| run_large_20251107_133202 | 1 | No | 1-stage |
| run_large_20251111_075600 | 5 | No | 1-stage |
| run_large_20251111_181836 | 5 | 10 | 1-stage |
| run_large_20251112_150502 | 22 | 10 | 1-stage |
| run_large_20260115_191350 | 5 | 10 | 2-stage |

### Molecule Models (3 variants)
| Checkpoint | Epochs | Concat | Architecture |
|------------|--------|--------|--------------|
| run_large_20251112_071557 | 5 | 10 | 1-stage |
| run_large_20251113_074900 | 5 | No | 1-stage |
| run_large_20260116_074355 | 5 | 10 | 2-stage |

## Setup

```bash
cd property_prediction
python -m venv venv_property
source venv_property/bin/activate
pip install -r requirements.txt
```

## Usage

1. **Extract features** (requires GPU):
   ```bash
   python scripts/extract_all_features.py
   ```

2. **Run experiments**:
   ```bash
   python scripts/run_all_experiments.py
   ```

3. **View results**: Check `experiments/` notebooks or `results/tables/`

## Directory Structure

```
property_prediction/
├── data/                    # Datasets
│   ├── polymer/            # PI1M_Tg_MAC.csv
│   └── molecule/           # lipophilicity.csv, bbbp.csv
├── featurizers/            # Feature extraction modules
├── models/                 # Prediction models (XGBoost)
├── experiments/            # Jupyter notebooks
├── results/                # Output
│   ├── features/          # Cached feature vectors
│   ├── models/            # Trained models
│   └── tables/            # Comparison tables
└── scripts/                # Automation scripts
```

## Citation

If using the PI1M_Tg_MAC dataset, cite:
- Zenodo: https://zenodo.org/records/17033425
- "A Closed-Loop Deep Generative Model for the Inverse Design of Radiation-Resistant Polymers"







