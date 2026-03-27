"""
Frozen ChemBERTa Embeddings Experiment
=======================================
Extract frozen (no fine-tuning) ChemBERTa embeddings for BBBP and HIV,
then run 5-fold stratified XGBoost. Same protocol as H-Net and RDKit.

Purpose: provide a fair language-model baseline for rebuttals.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / 'property_prediction' / 'data' / 'molecule'
RESULTS_DIR = PROJECT_ROOT / 'property_prediction' / 'results' / 'tables'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Imports ──────────────────────────────────────────────────────────────────
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
from tqdm import tqdm

# Device: MPS can conflict with XGBoost's libomp on macOS; use CPU to be safe
DEVICE = 'cpu'
print(f"Using device: {DEVICE}")


# ── ChemBERTa featurizer ──────────────────────────────────────────────────────
def load_chemberta(model_name='seyonec/ChemBERTa-zinc-base-v1'):
    print(f"Loading ChemBERTa: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M (frozen)")
    return tokenizer, model


def extract_chemberta_embeddings(smiles_list, tokenizer, model, batch_size=64):
    """Mean-pool last hidden state over non-padding tokens."""
    all_embeddings = []
    for i in tqdm(range(0, len(smiles_list), batch_size), desc="ChemBERTa embeddings"):
        batch = smiles_list[i:i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt',
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        # out.last_hidden_state: (B, L, D)
        hidden = out.last_hidden_state  # (B, L, 768)
        attn_mask = enc['attention_mask'].unsqueeze(-1).float()  # (B, L, 1)
        summed = (hidden * attn_mask).sum(dim=1)
        lengths = attn_mask.sum(dim=1).clamp(min=1)
        pooled = (summed / lengths).cpu().float().numpy()  # (B, 768)
        all_embeddings.append(pooled)
    return np.vstack(all_embeddings)


# ── XGBoost CV (matches paper protocol) ──────────────────────────────────────
def run_cv(X, y, n_folds=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    auc_scores, acc_scores = [], []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        clf = HistGradientBoostingClassifier(
            max_iter=500, max_depth=6, learning_rate=0.05,
            random_state=random_state,
        )
        clf.fit(X[tr_idx], y[tr_idx])
        y_prob = clf.predict_proba(X[val_idx])[:, 1]
        y_pred = clf.predict(X[val_idx])
        auc_scores.append(roc_auc_score(y[val_idx], y_prob))
        acc_scores.append(accuracy_score(y[val_idx], y_pred))
        print(f"  Fold {fold+1}: AUC={auc_scores[-1]:.4f}, Acc={acc_scores[-1]:.4f}")
    return np.array(auc_scores), np.array(acc_scores)


# ── Load datasets ─────────────────────────────────────────────────────────────
def load_bbbp():
    df = pd.read_csv(DATA_DIR / 'bbbp.csv')
    mask = df['smiles'].notna() & df['p_np'].notna()
    return df.loc[mask, 'smiles'].values, df.loc[mask, 'p_np'].values.astype(int)


def load_hiv(max_samples=None):
    df = pd.read_csv(DATA_DIR / 'hiv.csv')
    mask = df['smiles'].notna() & df['HIV_active'].notna()
    df = df[mask]
    if max_samples:
        df = df.sample(max_samples, random_state=42)
    return df['smiles'].values, df['HIV_active'].values.astype(int)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    tokenizer, model = load_chemberta()
    results = []

    # ── BBBP ──────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("BBBP (Blood-Brain Barrier Penetration)")
    print("="*60)
    smiles, y = load_bbbp()
    print(f"  Dataset: {len(smiles)} samples, class balance: {y.mean():.3f}")

    X = extract_chemberta_embeddings(smiles.tolist(), tokenizer, model)
    print(f"  Embeddings shape: {X.shape}")

    auc_scores, acc_scores = run_cv(X, y)
    bbbp_result = {
        'dataset': 'BBBP',
        'model': 'ChemBERTa (frozen)',
        'task': 'classification',
        'metric': 'AUC',
        'mean': float(auc_scores.mean()),
        'std': float(auc_scores.std()),
        'fold_scores': auc_scores.tolist(),
        'acc_mean': float(acc_scores.mean()),
        'acc_std': float(acc_scores.std()),
    }
    results.append(bbbp_result)
    print(f"\n  BBBP AUC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")

    # ── HIV ───────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("HIV (HIV Replication Inhibition)")
    print("="*60)
    smiles_hiv, y_hiv = load_hiv()
    print(f"  Dataset: {len(smiles_hiv)} samples, class balance: {y_hiv.mean():.3f}")

    X_hiv = extract_chemberta_embeddings(smiles_hiv.tolist(), tokenizer, model)
    print(f"  Embeddings shape: {X_hiv.shape}")

    auc_hiv, acc_hiv = run_cv(X_hiv, y_hiv)
    hiv_result = {
        'dataset': 'HIV',
        'model': 'ChemBERTa (frozen)',
        'task': 'classification',
        'metric': 'AUC',
        'mean': float(auc_hiv.mean()),
        'std': float(auc_hiv.std()),
        'fold_scores': auc_hiv.tolist(),
        'acc_mean': float(acc_hiv.mean()),
        'acc_std': float(acc_hiv.std()),
    }
    results.append(hiv_result)
    print(f"\n  HIV AUC: {auc_hiv.mean():.4f} ± {auc_hiv.std():.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / 'chemberta_frozen_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print("\n" + "="*60)
    print("SUMMARY (ChemBERTa frozen + XGBoost, 5-fold CV)")
    print("="*60)
    print(f"{'Dataset':<12} {'Model':<25} {'AUC mean':>10} {'AUC std':>10}")
    print("-"*60)
    for r in results:
        print(f"{r['dataset']:<12} {r['model']:<25} {r['mean']:>10.4f} {r['std']:>10.4f}")

    # Compare against paper values
    paper = {'BBBP': {'RDKit': 0.927, 'H-Net': 0.950}, 'HIV': {'RDKit': 0.760, 'H-Net': 0.788}}
    print("\nComparison with paper results (mean only):")
    print(f"{'Dataset':<12} {'RDKit':>10} {'H-Net':>10} {'ChemBERTa(frozen)':>20}")
    print("-"*55)
    for r in results:
        ds = r['dataset']
        print(f"{ds:<12} {paper[ds]['RDKit']:>10.3f} {paper[ds]['H-Net']:>10.3f} {r['mean']:>20.4f}")


if __name__ == '__main__':
    main()
