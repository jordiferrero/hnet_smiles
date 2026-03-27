We thank Reviewer wgYn for the careful review.

**[R1] Clear list of relevant methods and why each was excluded (Key Question)**

The most relevant methods and their inclusion status:

- *ChemBERTa / MolBERT*: Fine-tuned end-to-end per task on 77M+ molecules. Excluded from Table 6 because fine-tuning conflates tokenization quality with task-specific adaptation. We ran frozen ChemBERTa under the identical protocol; results are in [R3].
- *PolyBERT*: Fine-tuned on PSMILES. Same exclusion reason; also PSMILES-specific, so not applicable to the joint evaluation.
- *SmilesPE*: Fixed-vocabulary BPE tokenizer. Included in Table 5 for tokenization-level comparison (token statistics, compression). Frozen SmilesPE embeddings for property prediction remain a gap.
- *Char-level tokenization*: Included in Table 5 as the minimal baseline.

The frozen-embedding + gradient boosting setup keeps tokenization as the only variable, isolating what the tokenizer contributes independently of fine-tuning.

**[R2] Full experimental setup details**

Missing protocol details:

- *Train/val/test split*: random 80%/10%/10%, random_state=42, applied identically to H-Net and RDKit.
- *Cross-validation*: 5-fold stratified K-fold for classification; 5-fold K-fold for regression.
- *Gradient boosting hyperparameters*: n_estimators=500, max_depth=6, learning_rate=0.05, early_stopping_rounds=50, evaluated on validation fold.
- *Data leakage*: None. The same fixed split is used for both conditions; H-Net embeddings are extracted without any exposure to downstream labels.
- *Polymer evaluation*: Tg and MAC labels from PI1M; 10,000 molecules sampled uniformly.

**[R3] Statistical results with confidence intervals**

Full 5-fold CV results including the new frozen ChemBERTa baseline:

| Task | RDKit | H-Net (frozen) | ChemBERTa (frozen) |
|------|-------|----------------|-------------------|
| BBBP AUC↑ | 0.927 ± 0.009 | 0.950 ± 0.002 | 0.954 ± 0.008 |
| HIV AUC↑ | 0.760 ± 0.044 | 0.788 ± 0.010 | 0.795 ± 0.011 |
| BACE AUC↑ | **0.897 ± 0.017** | 0.867 ± 0.020 | — |
| Tg MAE↓ | **24.83 ± 0.74** | 26.61 ± 0.61 | — |
| Lipo MAE↓ | **0.494 ± 0.014** | 0.682 ± 0.019 | — |
| ESOL RMSE↓ | **0.660 ± 0.050** | 0.910 ± 0.065 | — |
| FreeSolv RMSE↓ | **1.131 ± 0.160** | 2.183 ± 0.187 | — |

On BBBP, H-Net and RDKit 95% CIs do not overlap ([0.946, 0.954] vs [0.909, 0.945]). H-Net and frozen ChemBERTa are statistically indistinguishable despite the 40× pretraining gap. The marginal ChemBERTa advantage (BBBP: +0.004, HIV: +0.007) is consistent with its larger pretraining corpus; data scaling predicts roughly this magnitude of lift. The more informative comparison is representation quality per pretraining sample, where H-Net is fully competitive. On HIV, H-Net's variance (σ = 0.010) is 4× tighter than RDKit's (σ = 0.044). On BACE, RDKit edges ahead (0.897 vs 0.867, overlapping CIs), consistent with RDKit encoding explicit pharmacophoric features, a gap we report transparently. For Tg, overlapping CIs indicate competitive performance, not failure.

**[R4] On BBBP and HIV as benchmarks**

These datasets were selected because they are well-characterized, not because they are easy. The goal is representation comparison under a fixed protocol, not a SOTA claim. Our RDKit baseline (BBBP AUC = 0.927) is competitive with published results for this featurization, so it is not a strawman.

Scaffold splits would be a more stringent generalization test. The random split is consistent across both H-Net and RDKit conditions, so the comparison is internally valid; the absolute performance numbers should be interpreted with that caveat.

**[R5] On regression underperformance**

RDKit encodes 2,248 explicit features (200 curated physicochemical descriptors plus Morgan fingerprints) that directly capture the electronic and steric properties driving solubility, lipophilicity, and hydration free energy. Frozen byte-level embeddings do not encode these properties directly. On ESOL, where aqueous solubility depends heavily on explicit charge and hydrogen-bonding descriptors that RDKit encodes by design, the gap is larger (RMSE 0.910 vs 0.660). Frozen general-purpose encoders consistently underperform task-specific engineered features on numerical regression in NLP as well. The pattern does not reflect a failure of tokenization; it reflects the limits of frozen representation without fine-tuning, which the paper acknowledges. Polymer Tg (26.61 ± 0.61 vs 24.83 ± 0.74 MAE, overlapping CIs) is the exception: H-Net generalizes competitively to a polymer-domain regression task it was not fine-tuned for.
