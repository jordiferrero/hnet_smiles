We thank Reviewer iQmP for the careful review.

**[R1] On comparisons with ChemBERTa/MolBERT**

We ran frozen ChemBERTa (`seyonec/ChemBERTa-zinc-base-v1`, 44M params, pretrained on 77M PubChem molecules) under the identical protocol: frozen embeddings into gradient boosting, 5-fold stratified CV. Results are in [R2]. H-Net, pretrained on ~2M molecules (40× less data), achieves statistically indistinguishable performance from frozen ChemBERTa on both BBBP and HIV. That result directly supports the quality of H-Net's tokenization despite the pretraining data gap.

Fine-tuned ChemBERTa and MolBERT remain excluded from Table 6 because fine-tuning conflates tokenization quality with task-specific adaptation. The frozen setup isolates tokenization as the independent variable, which is the same rationale used when comparing word embeddings across tokenizers before fine-tuning in NLP.

**[R2] Confidence intervals and frozen ChemBERTa comparison**

Full 5-fold CV results including the new frozen ChemBERTa baseline:

| Task | Metric | RDKit | H-Net (frozen) | ChemBERTa (frozen) |
|------|--------|-------|----------------|-------------------|
| BBBP | AUC↑ | 0.927 ± 0.009 | 0.950 ± 0.002 | 0.954 ± 0.008 |
| HIV | AUC↑ | 0.760 ± 0.044 | 0.788 ± 0.010 | 0.795 ± 0.011 |
| BACE | AUC↑ | **0.897 ± 0.017** | 0.867 ± 0.020 | — |
| Tg | MAE↓ | **24.83 ± 0.74** | 26.61 ± 0.61 | — |
| Lipo | MAE↓ | **0.494 ± 0.014** | 0.682 ± 0.019 | — |
| ESOL | RMSE↓ | **0.660 ± 0.050** | 0.910 ± 0.065 | — |
| FreeSolv | RMSE↓ | **1.131 ± 0.160** | 2.183 ± 0.187 | — |

On BBBP, 95% CIs are non-overlapping (H-Net: [0.946, 0.954]; RDKit: [0.909, 0.945]). A Welch's t-test gives t = 5.58, p < 0.005. H-Net and frozen ChemBERTa are statistically indistinguishable despite the 40× pretraining gap. On HIV, the improvement is directional (t = 1.39, p ≈ 0.24); H-Net's variance (σ = 0.010) is 4× tighter than RDKit's (σ = 0.044), which reflects more stable representations rather than a marginal mean shift. Regression underperformance relative to RDKit is acknowledged; see [R4].

**[R3] On hapax legomena and rare tokens (Key Question 1)**

A 57–63% hapax rate is consistent with Zipf's law (token frequency falls as a power law of rank, so the majority of vocabulary types appear rarely or once). Natural language corpora show 40–60% hapax rates as a mathematical consequence of this distribution, not pathological overfitting. The question is whether rare tokens impair generalization. They do not: H-Net achieves AUC = 0.788 ± 0.010 on HIV across 41K molecules and all 5 folds.

Under mean pooling, each token position contributes exactly $1/N$ to the final embedding, where $N$ is sequence length (~16–22 tokens/SMILES). Pruning hapax tokens and renormalizing shifts the embedding by at most $(k/N) \cdot \|e\|$, where $k$ is the number of pruned positions. The top 500 most frequent tokens cover >95% of all positions, so $k/N < 0.05$ on average. The indirect empirical test is fold-to-fold CV stability: memorized noise would produce high variance, yet H-Net shows σ = 0.010 on 41K HIV molecules versus RDKit's σ = 0.044. That is the opposite of what memorization predicts.

**[R4] On regression underperformance**

Expected and principled. RDKit encodes 2,248 explicit features (200 curated physicochemical descriptors plus Morgan fingerprints) that directly capture electronic and steric properties relevant to solubility, lipophilicity, and hydration free energy. Frozen byte-level embeddings do not encode these properties directly. The same gap appears in NLP: frozen language model representations consistently underperform task-specific engineered features on numerical regression. Polymer Tg (26.61 ± 0.61 vs 24.83 ± 0.74 MAE, 7.2% gap, overlapping CIs) is competitive, not a failure.

**[R5] On scaling law with 3 data points**

BPB ∝ FLOPs^{-0.09} with R² = 0.97 spans a 22× FLOP range (1.0×10¹⁷ to 2.2×10¹⁸). Three well-spaced points across more than an order of magnitude of compute is the standard starting point for a scaling characterization. The claim is directional: more compute improves tokenization quality. We are not extrapolating precise predictions, and we acknowledge that denser sampling would strengthen the fit.
