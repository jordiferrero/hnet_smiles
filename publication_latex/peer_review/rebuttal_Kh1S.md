We thank Reviewer Kh1S for the precise review.

**[R1] On the "static" vs "dynamic" distinction and SmilesPE**

We accept this as a partial correction. SmilesPE is more accurately described as "data-driven fixed-vocabulary": it trains BPE on a fixed corpus (ChEMBL) to produce a vocabulary, applied without modification at inference. H-Net works differently: boundary probabilities are computed _contextually_ from each input sequence at inference time, with no pre-computed vocabulary. The tokenization behavior evolves with training and adapts to local sequence context. H-Net has no vocabulary and never freezes one. That is the architectural distinction, regardless of what "dynamic" is taken to mean.

"Fixed-vocabulary" is the more precise term, and we accept the correction.

**[R2] On the expected nature of dataset-specific vocabularies**

Domain-specific vocabularies are expected under SmilesPE too. Agreed. But the paper's contribution is the quantification, not the observation: KL divergence = 3.92, Jaccard = 0.30 for non-concatenated models and 0.06 for concatenated, achieved without any manually designed chemical vocabulary. The finding that H-Net respects atom boundaries in 70% of cases without chemical supervision was not known before these experiments. Neither was the systematic relationship between tokenization behavior and compute, architecture, and concatenation strategy. These are not obvious consequences of applying BPE to chemistry; they are specific properties of end-to-end learned dynamic tokenization that had not been characterized.

**[R3] On comparison with atom-level tokenization and SmilesPE in property prediction**

Table 5 gives direct tokenization-level comparison with character-level and SmilesPE baselines. On downstream prediction, we ran frozen ChemBERTa (`seyonec/ChemBERTa-zinc-base-v1`, 44M params, pretrained on 77M PubChem molecules) under the identical frozen-embedding + gradient boosting protocol:

| Task      | RDKit         | H-Net (frozen) | ChemBERTa (frozen) |
| --------- | ------------- | -------------- | ------------------ |
| BBBP AUC↑ | 0.927 ± 0.009 | 0.950 ± 0.002  | 0.954 ± 0.008      |
| HIV AUC↑  | 0.760 ± 0.044 | 0.788 ± 0.010  | 0.795 ± 0.011      |

Both frozen language model approaches outperform RDKit. H-Net and frozen ChemBERTa are statistically indistinguishable despite the 40× pretraining data gap. Frozen SmilesPE embeddings for property prediction remain a gap in the current evaluation.

**[R4] On SMILES randomization**

PI1M and MOSES use canonical SMILES, consistent with their dataset conventions. SMILES randomization augmentation would change training dynamics independently of tokenization strategy, making it harder to isolate the tokenization effect. We agree this is a real limitation: augmentation could reduce sensitivity to canonical ordering and is an important direction for follow-up work.

**[R5] On generative tasks**

Molecular generation is the most natural downstream application of an expressive tokenizer and is already identified in the conclusion as a priority. This paper asks whether dynamic tokenization works for chemistry at all, and what it learns; that question has to come before using it generatively. The characterization here is the prerequisite.

**[R6] On missing related work**

1. _"Exploring data-driven chemical SMILES tokenization approaches to identify key protein-ligand binding moieties"_ (Mol. Informatics, 2024): directly related to our motivation. We cite Li & Fourches (2021, SmilesPE) as the foundational data-driven chemical tokenization reference; this 2024 work is a natural companion citation we acknowledge was missing.

2. _fragSMILES_ (Comm. Chem., 2025): a notation-level approach, not a tokenization-level one. Our work tokenizes standard SMILES; fragSMILES is complementary, and one could apply dynamic tokenization to fragSMILES strings directly.

**[R7] On whether the key findings are unique to dynamic tokenization**

The vocabulary divergence is verified quantitatively for the first time, achieved without any manually designed chemical vocabulary. That is what separates H-Net from SmilesPE: one model adapts across SMILES and PSMILES domains without separate tokenizer engineering. Crucially, SmilesPE cannot exhibit these behaviors by construction: its vocabulary is frozen at training time, so vocabulary drift with context length, architecture, and domain is not observable. The Jaccard dropping from 0.30 (non-concatenated) to 0.06 (concatenated) reveals structure in chemical tokenization that was previously unmeasurable, not merely unobserved.
