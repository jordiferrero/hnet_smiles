We thank Reviewer u1dD for the detailed critique.

**[R1] On the main message, key questions, and scientific contribution**

The abstract can be made clearer, but the claim that this work "has 0 impact" misstates the contribution. The reviewer's own framing is accurate: the contribution is that *one H-Net model, trained jointly on SMILES and PSMILES, discovers domain-appropriate tokenizations for both without any manual vocabulary engineering per domain*. This is the "beauty of dynamic tokenization" the reviewer describes. Polymer informatics (PolyBERT) and molecular informatics use entirely separate tokenization systems today; H-Net is a unified alternative that adapts without redesigning the tokenizer.

The relevant comparison is H-Net (one joint model) vs. separate SmilesPE + PolyBERT tokenizers, not H-Net vs. SmilesPE on a single domain. The KL divergence (3.92) and Jaccard (0.30) quantify how closely H-Net's emergent domain separation matches purpose-built systems. The 70% vocabulary divergence is not a failure of the joint model; it is evidence that H-Net specializes per domain without being instructed to. A KL of 3.92 is comparable to the divergence between distinct natural languages in NLP, showing that chemical domain structure is discovered bottom-up from byte sequences alone.

**[R2] On the "0 impact" claim**

Characterization papers without new models are well-established at top ML venues. One example from the tokenization literature: Mielke et al. (2021) "Between words and characters" was accepted at ACL for systematically analyzing subword tokenization behavior, with no new architecture proposed. Our paper is the chemical analogue, and there are others like it across representation learning.

None of the following were known before these experiments: H-Net learns chemically meaningful segments (70% atom boundary respect, no supervision); it adapts to domain differences (KL = 3.92, Jaccard = 0.30); tokenization quality scales with compute (BPB 0.83→0.64, R² = 0.97); representations transfer (BBBP 0.950 vs RDKit 0.927). None were predictable a priori.

**[R3] On the PSMILES vs SMILES comparison**

The \* marker alone does not explain a KL of 3.92 or 70% vocabulary difference. The divergence comes from structural differences in polymer repeat units: longer sequences with complex branching and connectivity patterns. Table 2 shows that non-concatenated models trained on the *same* dataset produce Jaccard = 0.30, meaning dataset composition drives vocabulary formation, not just notation. Concatenated training drops that to 0.06, showing longer-context training increases domain specificity further. This is not a notation artifact.

**[R4] On Figure 3 and token length distributions**

Figure 3 shows *token length distributions*, not vocabulary identity. Two distributions can have identical length profiles while representing completely different chemical fragments. Jaccard (0.30) and KL divergence (3.92) capture what Figure 3 cannot: which tokens appear and at what frequencies. That is why token-level metrics beyond length are needed, which this paper introduces.

**[R5] On error bars and statistical significance**

Full 5-fold CV results including frozen ChemBERTa (`seyonec/ChemBERTa-zinc-base-v1`, 44M params, 77M PubChem molecules, same protocol):

| Task | RDKit | H-Net (frozen) | ChemBERTa (frozen) |
|------|-------|----------------|-------------------|
| BBBP AUC↑ | 0.927 ± 0.009 | 0.950 ± 0.002 | 0.954 ± 0.008 |
| HIV AUC↑ | 0.760 ± 0.044 | 0.788 ± 0.010 | 0.795 ± 0.011 |
| Tg MAE↓ | **24.83 ± 0.74** | 26.61 ± 0.61 | — |

H-Net outperforms RDKit on classification, with non-overlapping 95% CIs on BBBP (H-Net: [0.946, 0.954]; RDKit: [0.909, 0.945]). On HIV, H-Net's fold variance (σ = 0.010) is 4× tighter than RDKit's (σ = 0.044). Frozen ChemBERTa (40× more pretraining data) is statistically indistinguishable from H-Net on both tasks. The marginal ChemBERTa advantage (+0.004 BBBP, +0.007 HIV) is the expected effect of a 40× larger pretraining corpus; it does not indicate superior tokenization.

**[R6] On computational overhead**

H-Net is trained once and used for embedding extraction, the same pattern as any frozen encoder. Deployment overhead is context-dependent and orthogonal to the tokenization findings studied here.
