# ICML 2026 Paper Draft: Dynamic Tokenization for Chemical SMILES

## Proposed Title Options

**Primary:** *Learning Chemical Grammar: Dynamic Tokenization for SMILES Representation with Hierarchical Networks*

**Alternatives:**
- *Beyond Static Tokenizers: Adaptive Byte-Level Chunking for Molecular and Polymer SMILES*
- *H-Net for Chemistry: Learning Dataset-Specific Token Vocabularies from SMILES Strings*
- *From Bytes to Bonds: Dynamic Tokenization as Foundation for Chemical Language Models*

---

## Abstract (250 words max)

**Draft:**

Tokenization fundamentally shapes how machine learning models represent chemical structures. While static tokenizers like SMILES Pair Encoding (SmilesPE) use predefined subword vocabularies, we investigate *dynamic tokenization* using Hierarchical Networks (H-Net), which learn byte-level chunking patterns directly from data. We train 350M-parameter H-Net models on polymeric (PI1M) and molecular (MOSES) SMILES datasets, systematically analyzing how dataset nature, training context (concatenation), training duration, and architectural choices affect learned tokenization.

Our observability study reveals three key findings: (1) **Dataset specificity matters** – H-Net learns distinct "chemical vocabularies" with only 30% token overlap between polymer and molecular datasets, analogous to language-specific patterns in NLP; (2) **Dynamic tokenization improves with training** – extended training (22 epochs) yields 63% more unique tokens and 23% higher efficiency (fewer tokens per SMILES), achieving bits-per-byte (BPB) of 0.64, comparable to well-trained language models; (3) **Learned representations transfer** – H-Net embeddings serve as effective foundation model features, outperforming RDKit descriptors on blood-brain barrier penetration classification (AUC 0.95 vs 0.93) and achieving competitive performance on polymer glass transition temperature prediction (MAE 26.6°C vs 24.8°C).

We compare H-Net's fine-grained byte-level approach (2-3 character tokens, 6-8K vocabulary) against SmilesPE's coarse-grained subwords (4-6 character tokens, 1.6K vocabulary), demonstrating complementary strengths. Our work establishes dynamic tokenization as a promising paradigm for chemical representation learning, with implications for molecular foundation models.

---

## 1. Introduction (~1 page)

### 1.1 Opening Hook & Motivation
- **The tokenization bottleneck**: How molecules are tokenized fundamentally limits what patterns models can learn
- Static tokenizers (SMILES, SELFIES, SmilesPE, BPE) impose fixed vocabularies → may miss dataset-specific patterns
- Parallel to NLP: language-specific tokenizers outperform universal ones

### 1.2 Research Gap
- Prior work: Fixed tokenization schemes optimized for general chemistry
- **Question**: Can *learned*, dynamic tokenization adapt to specific chemical domains (polymers vs. small molecules)?
- No systematic study of how training variables affect tokenization of chemical strings

### 1.3 Our Contributions (Bullet points)
1. **Systematic observability study** of dynamic tokenization for chemical SMILES using H-Net (6 1-stage + 2 2-stage models, 350M parameters each)
2. **Four-way analysis**: Dataset nature (polymer vs. molecular), concatenation, training amount, architecture (1-stage vs. 2-stage)
3. **Quantitative metrics suite**: Token statistics, Jaccard similarity, KL divergence, breakpoint analysis, bits-per-byte compression
4. **Foundation model validation**: H-Net embeddings as chemical featurizers for property prediction
5. **Benchmark comparison**: Systematic comparison with industry-standard SmilesPE

### 1.4 Paper Organization
- Brief roadmap of remaining sections

---

## 2. Related Work (~1 page)

### 2.1 Chemical String Representations
- SMILES: De facto standard, limitations (non-unique, arbitrary ordering)
- SELFIES: 100% valid molecules, but verbose
- PSMILES: Polymer-specific notation

### 2.2 Tokenization Strategies for Chemistry
- **Character-level**: Simple but loses chemical meaning
- **Atom-level**: Chemistry-aware but fixed vocabulary
- **Subword methods**: BPE, SmilesPE (Pair Encoding from ChEMBL)
- Gap: All above are *static* – same tokenization regardless of dataset

### 2.3 Learned Tokenization
- **H-Net (Hwang et al., 2025)**: Hierarchical networks with dynamic chunking
  - Byte-level processing with learned boundaries
  - Originally for NLP: different tokens for Mandarin vs. English
  - **Our contribution**: First application to chemical SMILES
- BLT (Meta): Byte-level transformers
- Neural tokenization in vision (ViT patches)

### 2.4 Chemical Language Models & Foundation Models
- ChemBERTa, MolBERT: Masked language modeling
- MolGPT, ChemGPT: Autoregressive generation
- SMI-TED: Encoder-decoder for molecules
- **Gap**: Focus on downstream tasks, not tokenization analysis

---

## 3. Methods (~1.5 pages)

### 3.1 H-Net Architecture Overview
- **Architecture layout**: `["m4", ["T22"], "m4"]` (1-stage) or `["m4", ["T1m4", ["T22"], "m4T1"], "m4"]` (2-stage)
- **Components**:
  - Mamba blocks (m4): State space models for sequence processing
  - Transformer blocks (T22): Attention with boundary prediction
  - Hierarchical learning rate modulation: Outer stages 3×, core 0.9×
- **Tokenization mechanism**: Byte-level input → learned boundary prediction → dynamic chunks
- **Parameters**: ~350M, d_model=[1024, 1536], vocab_size=256 (byte-level)

### 3.2 Training Configuration
- **Datasets**: 
  - PI1M: ~1M polymeric SMILES (PSMILES)
  - MOSES: Molecular SMILES (drug-like molecules)
- **Concatenation**: 10 SMILES per training example (optional)
- **Loss**: Cross-entropy + 0.01 × load balancing loss
- **Optimization**: AdamW, LR=1e-4, gradient accumulation=8
- **Training duration**: 1, 5, or 22 epochs

### 3.3 Experimental Matrix

| Variable | Values Tested |
|----------|--------------|
| Dataset | PI1M (polymer), MOSES (molecular) |
| Concatenation | None, 10× SMILES |
| Epochs | 1, 5, 22 |
| Architecture | 1-stage, 2-stage |

**Total models**: 8 H-Net (6 × 1-stage + 2 × 2-stage)

### 3.4 Analysis Metrics
- **Token statistics**: Total tokens, unique tokens, tokens per SMILES, mean token length
- **Distribution comparison**: Jaccard similarity (vocabulary overlap), KL divergence (frequency distributions)
- **Breakpoint analysis**: Character positions where tokens split
- **Compression**: Bits-per-byte (BPB = CE_loss / ln(2)), Perplexity (PPL = exp(CE_loss))

### 3.5 Benchmark: SmilesPE
- Pre-trained vocabulary from ChEMBL (~30K tokens)
- Subword-level tokenization (4-6 character average)
- Same evaluation on PI1M and MOSES

### 3.6 Property Prediction Setup (Foundation Model Validation)
- **Feature extraction**: Mean pooling of H-Net hidden states (768-dim)
- **Baseline**: RDKit descriptors (Morgan fingerprints + 200 physicochemical features)
- **Predictor**: XGBoost with 5-fold cross-validation
- **Tasks**:
  - Polymer: Tg (regression), MAC (regression)
  - Molecule: Lipophilicity (regression), BBBP (classification)

---

## 4. Results (~2 pages) — MAIN FOCUS OF PAPER

### 4.1 Dataset Nature Effect: Polymers vs. Molecules
**Key finding: H-Net learns distinct chemical vocabularies**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Token Overlap (Jaccard) | 30-33% | Distinct vocabularies |
| Breakpoint Overlap | 57-59% | Some shared chemical motifs |
| KL Divergence | 3.9-4.3 | Large distribution difference |
| Tokens/SMILES difference | +20% for polymers | Polymer complexity |

**Interpretation**: Like learning different tokens for Mandarin vs. English, H-Net discovers dataset-specific "chemical languages" for polymeric vs. molecular structures.

**Figures**: 
- Token length distributions (polymer vs. molecular)
- Top-50 tokens comparison heatmap

### 4.2 Concatenation Effect
**Key finding: Concatenation creates specialized tokens while preserving core logic**

| Dataset | Token Overlap | Breakpoint Stability |
|---------|---------------|----------------------|
| Polymer (PI1M) | 35% | 76% |
| Molecular (MOSES) | 45% | **96%** |

**Surprising result**: Molecular SMILES show remarkable stability (96% breakpoint agreement), while polymers change more. Hypothesis that polymers (inherently concatenated) would be less affected was NOT supported.

### 4.3 Training Amount Effect
**Key finding: More training = more efficient, specialized tokenization**

| Epochs | Unique Tokens | Tokens/SMILES | Mean Token Length | BPB |
|--------|---------------|---------------|-------------------|-----|
| 1 | 4,903 | 21.6 | 2.20 | 0.83 |
| 5 | 5,775 (+18%) | 18.2 (-16%) | 2.62 (+19%) | 0.69 |
| 22 | 8,019 (+39%) | 16.6 (-9%) | 2.87 (+10%) | **0.64** |
| **Total** | **+63%** | **-23%** | **+30%** | **-23%** |

**Key insight**: Extended training doesn't change *where* tokens split (82-91% breakpoint overlap) but *what* specialized tokens are learned. Model discovers increasingly efficient representations.

### 4.4 Compression Metrics (BPB & Perplexity)
**Key finding: H-Net learns meaningful chemical "grammar"**

- Best BPB: **0.64** (22 epochs) — far better than random (8.0), comparable to English LMs (~1.0-1.5)
- Perplexity: 1.56-1.61 across models
- Confirms H-Net learns predictable chemical patterns (functional groups, rings, stereochemistry)

### 4.5 Architecture Effect: 1-Stage vs. 2-Stage
**Key finding: 2-stage provides marginal compression improvement with added interpretability**

| Dataset | 1-Stage BPB | 2-Stage BPB | Improvement |
|---------|-------------|-------------|-------------|
| PI1M | 0.687 | 0.686 | +0.2% |
| MOSES | 0.682 | 0.679 | +0.4% |

2-stage learns hierarchical chunking (bytes → chunks → super-chunks) but improvement is minimal. Trade-off: more structure, less vocabulary diversity.

### 4.6 H-Net vs. SmilesPE Benchmark
**Key finding: Complementary approaches with distinct philosophies**

| Aspect | H-Net | SmilesPE |
|--------|-------|----------|
| Token length | 2-3 chars (byte-level) | 4-6 chars (subwords) |
| Vocabulary | 6-8K (adaptive) | 1.6-2K (fixed) |
| Tokens/SMILES | 16-22 (fine-grained) | 6-11 (compressed) |
| **Advantage** | Dataset-adaptive, continuously improves | Interpretable, efficient |

**Figure**: Side-by-side tokenization example

---

## 5. Application: H-Net as Foundation Model (~1 page) — SUPPORTING EVIDENCE

### 5.1 Property Prediction Results

#### 5.1.1 Polymer Tasks (PI1M)
| Task | RDKit MAE | H-Net MAE | Gap |
|------|-----------|-----------|-----|
| **Tg (°C)** | 24.8 | 26.6 | +7% |
| MAC | 5.7e-5 | 10.3e-5 | +81% |

H-Net competitive on Tg (within 2°C of RDKit, comparable to SOTA Lieconv-Tg at 24.4K)

#### 5.1.2 Molecule Tasks (MoleculeNet)
| Task | RDKit | H-Net | Winner |
|------|-------|-------|--------|
| Lipophilicity (MAE) | 0.494 | 0.682 | RDKit |
| **BBBP (AUC)** | 0.927 | **0.950** | **H-Net** |

H-Net outperforms RDKit on BBBP classification by +2.3% AUC

### 5.2 Critical Insights
- **Mean pooling essential**: CLS pooling fails catastrophically (AUC drops 26%)
- **Less training sometimes better**: 1-epoch H-Net often outperforms 22-epoch for downstream tasks
- **Task-dependent**: H-Net better for classification, RDKit for precise regression

### 5.3 Implications for Foundation Models
- H-Net embeddings capture structural information transferable to property prediction
- Complementary to traditional descriptors → potential for ensemble approaches

---

## 6. Discussion (~0.75 pages)

### 6.1 Why Does Dynamic Tokenization Work for Chemistry?
- Chemical strings have domain-specific patterns (functional groups, rings, stereochemistry)
- Static tokenizers impose arbitrary boundaries → lose chemical meaning
- Learned tokenization discovers *dataset-relevant* patterns

### 6.2 Trade-offs: H-Net vs. Static Tokenizers
| Consideration | H-Net | Static (SmilesPE) |
|---------------|-------|-------------------|
| Adaptability | ✓ Dataset-specific | Same for all |
| Interpretability | Harder (byte-level) | ✓ Chemical substructures |
| Compression | More tokens | ✓ Fewer tokens |
| Setup | Requires training | ✓ Zero-shot |

### 6.3 Practical Recommendations
- **Use H-Net when**: Working with novel chemical domains, need dataset-specific representations, have compute for training
- **Use SmilesPE when**: Need interpretable tokens, quick prototyping, standardized vocabularies

### 6.4 Limitations
- Training cost: 1-3 days per model on GPU
- Less interpretable than chemistry-aware tokenizers
- Byte-level processing increases sequence length

---

## 7. Conclusion (~0.5 pages)

### 7.1 Summary of Contributions
1. First systematic study of dynamic tokenization for chemical SMILES
2. Demonstrated dataset-specific vocabulary learning (30% overlap polymer vs. molecular)
3. Quantified training effects: 63% more unique tokens, 23% efficiency gain
4. Validated H-Net embeddings for property prediction (BBBP AUC 0.95)
5. Established complementary relationship with SmilesPE

### 7.2 Future Directions
- Hybrid approaches combining H-Net adaptability with chemistry-aware priors
- Interpretability analysis of learned tokens (what chemical patterns emerge?)
- Multi-dataset pretraining for chemical foundation models
- Integration with molecular generation pipelines

---

## Appendix (Unlimited pages)

### A. Full Training Configurations
- JSON configs for all 8 models
- Hyperparameter tables

### B. Complete Results Tables
- All metrics for all models
- Property prediction detailed results

### C. Additional Visualizations
- Compression dynamics (BPB vs. training bytes)
- All token length distributions
- Training loss curves

### D. Tokenization Examples
- Side-by-side tokenization of 10 representative molecules
- Polymer vs. molecular examples

### E. SmilesPE Baseline Details
- Vocabulary statistics
- Tokenization algorithm

---

## Key Figures for Main Paper (8 pages)

1. **Figure 1**: Architecture overview (H-Net with dynamic chunking)
2. **Figure 2**: Token vocabulary overlap heatmap (polymer vs. molecular vs. SmilesPE)
3. **Figure 3**: Training progression (unique tokens, efficiency, BPB vs. epochs)
4. **Figure 4**: H-Net vs. SmilesPE tokenization comparison
5. **Figure 5**: Property prediction results (bar chart comparing H-Net vs. RDKit)

---

## References (Key Citations)

1. Hwang et al. (2025) - H-Net: Dynamic Chunking with Hierarchical Networks
2. Li et al. - SmilesPE: SMILES Pair Encoding
3. Weininger - SMILES notation
4. MoleculeNet - Benchmark datasets
5. Ross et al. - ChemBERTa
6. Zhang et al. - Lieconv-Tg (polymer Tg SOTA)

---

*Draft prepared for ICML 2026 submission*
*Page estimate: Main paper ~8 pages + references + appendix*


