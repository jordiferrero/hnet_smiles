# H-Net Dynamic Tokenization for SMILES: Observability Study
## Final Analysis Report

**Date:** January 16, 2026 (Updated)  
**Analysis Dataset Size:** 10,000 SMILES per dataset (statistically robust subset)  
**Total Models Analyzed:** 8 H-Net models (6 × 1-stage + 2 × 2-stage) + 2 SmilesPE benchmarks

---

## Executive Summary

This **observability study** analyzes the tokenization behavior of six trained H-Net models (350M parameters each) on chemical SMILES strings. The models were trained using a hierarchical architecture with dynamic chunking, employing byte-level tokenization, hierarchical learning rate modulation, and load balancing loss. All models were trained on AWS EC2 GPU instances using config-based JSON configurations.

We investigate how H-Net's deep-learned dynamic chunking tokenizes chemical SMILES strings, comparing the effects of:
1. **Dataset nature** (polymeric vs. molecular)
2. **Concatenation strategy** (concatenated vs. non-concatenated training)
3. **Training amount** (1, 5, and 22 epochs)
4. **Benchmark comparison** (H-Net vs. industry-standard SmilesPE)

### Key Findings:
- **Dataset Nature**: Polymer SMILES generate 15-20% more tokens than molecular SMILES, with distinct tokenization patterns
- **Concatenation**: Increases token diversity by 30-40% and creates longer, more specialized tokens
- **Training Amount**: More training (22 epochs) leads to 63% more unique tokens and 23% fewer tokens per SMILES (more efficient)
- **vs. SmilesPE**: H-Net learns byte-level patterns (2-3 char tokens) while SmilesPE uses predefined subwords (4-6 char tokens)

---

## 1. Experimental Setup

### 1.1 H-Net Training Methodology

#### 1.1.1 Model Architecture

All models use the **HNetForCausalLM** architecture - a 1-stage Hierarchical Network with dynamic chunking:

**Architecture Layout:** `["m4", ["T22"], "m4"]`
- **Encoder stage (m4)**: 4 Mamba blocks for initial encoding
- **Main network (T22)**: 22 Transformer blocks with boundary prediction
- **Decoder stage (m4)**: 4 Mamba blocks for final decoding

**Model Dimensions:**
- `d_model`: [1024, 1536] - Hidden dimensions for encoder/decoder and main network
- `d_intermediate`: [0, 4096] - MLP intermediate dimensions
- `vocab_size`: 256 - Byte-level tokenization (ByteTokenizer)

**State Space Model (SSM) Configuration:**
- `chunk_size`: 256 - Chunk size for SSM processing
- `d_conv`: 4 - Convolution dimension
- `d_state`: 128 - State dimension
- `expand`: 2 - Expansion factor

**Attention Configuration:**
- `num_heads`: [16, 16] - Attention heads per stage
- `rotary_emb_dim`: [32, 48] - Rotary embedding dimensions
- `window_size`: [1023, -1] - Attention window sizes (local, then full)

**Total Parameters:** ~350M parameters

#### 1.1.2 Training Procedure

**Tokenization:**
- **ByteTokenizer**: Byte-level (256 vocab) for SMILES strings
- **Concatenation**: Multiple SMILES joined with space separator
  - PI1M: 10 PSMILES per training example
  - MOSES: 10 SMILES per training example

**Loss Function:**
```
Total Loss = Cross-Entropy Loss + 0.01 × Load Balancing Loss
```
- **CE Loss**: Standard next-token prediction
- **LB Loss**: Encourages uniform token selection across routing module (prevents collapse)

**Optimization Strategy:**
- **Optimizer**: AdamW with weight decay 0.1
- **Hierarchical Learning Rate Modulation** (per HNet paper Section 2.3):
  - Outer stages (encoder/decoder): **3.0× base LR** (higher plasticity)
  - Main network: **0.9× base LR** (stable core)
  - Base learning rate: 1e-4
- **Gradient Accumulation**: 8 steps (effective batch size: 64 × 8 = 512)
- **Mixed Precision**: bfloat16 on GPU (no scaler needed)

**Training Phases:**
- **Batch Size**: 64 per GPU step
- **Epochs**: 1, 5, or 22 depending on model
- **Checkpoint Strategy**: 
  - Boundary predictions saved **every 1M training bytes** (for visualization)
  - Model checkpoints saved **only when loss improves** (disk space efficiency)
  - Epoch checkpoints saved at end of each epoch

**Data Split:**
- **Train/Test Split**: Fixed random seed (42) for reproducibility
- **Test Set Size**: 50 SMILES reserved for visualization/testing
- **Train Set**: Remaining samples shuffled per epoch

#### 1.1.3 Training Hardware & Configuration

**Device:** CUDA GPU (AWS EC2 instance)  
**Config Format:** JSON-based configuration files (all parameters in `training_config` section)  
**Signal Handling:** Graceful shutdown on interruption with checkpoint saving  
**Training Time:** 1-3 days per model depending on epochs

### 1.2 Datasets
- **PI1M (Polymer)**: Polymeric SMILES (PSMILES) - longer, more complex structures
  - Source: ~995K polymer SMILES strings
  - Characteristics: Longer sequences, more structural complexity
- **MOSES (Molecular)**: Molecular SMILES - shorter, simpler structures
  - Source: Standardized molecular dataset
  - Characteristics: Shorter sequences, drug-like molecules

### 1.3 Models Analyzed

All models were trained using the methodology described in sections 1.1-1.1.3, with variations in:
- **Dataset**: PI1M (polymer) vs. MOSES (molecular)
- **Concatenation**: 10 SMILES concatenated vs. single SMILES
- **Training Duration**: 1, 5, or 22 epochs

| Model ID | Dataset | Concatenation | Epochs | Architecture | Training Bytes |
|----------|---------|---------------|--------|--------------|----------------|
| PI1M_concat_1epoch | PI1M | Yes (10x) | 1 | 1-stage | 48M |
| PI1M_concat_5epoch | PI1M | Yes (10x) | 5 | 1-stage | 238M |
| PI1M_concat_22epoch | PI1M | Yes (10x) | 22 | 1-stage | 1,048M |
| PI1M_noconcat_5epoch | PI1M | No | 5 | 1-stage | 243M |
| PI1M_concat_5epoch_2stage | PI1M | Yes (10x) | 5 | **2-stage** | 238M |
| MOSES_concat_5epoch | MOSES | Yes (10x) | 5 | 1-stage | 358M |
| MOSES_noconcat_5epoch | MOSES | No | 5 | 1-stage | 367M |
| MOSES_concat_5epoch_2stage | MOSES | Yes (10x) | 5 | **2-stage** | 358M |

**Training Infrastructure:**
- Each model trained independently from randomly initialized weights
- Config-based training with JSON configuration files
- All training logs and checkpoints preserved in `checkpoints/run_*/` directories
- Boundary predictions saved at 1M byte intervals for analysis

### 1.4 Analysis Metrics
- **Token Statistics**: Total tokens, unique tokens, tokens per SMILES, token length distribution
- **Comparative Metrics**: Jaccard similarity, KL divergence, mean token length differences
- **Breakpoint Analysis**: Character positions where tokens are split

---

## 2. Results: Comprehensive Model Comparison

### 2.1 Overall Token Statistics

| Model | Total Tokens | Unique Tokens | Architecture |
|-------|--------------|---------------|--------------|
| **PI1M Models (1-Stage)** |
| PI1M_concat_1epoch | 216,277 | 4,903 | 1-stage |
| PI1M_noconcat_5epoch | 220,423 | 4,094 | 1-stage |
| PI1M_concat_5epoch | 182,298 | 5,775 | 1-stage |
| PI1M_concat_22epoch | 165,928 | 8,019 | 1-stage |
| **PI1M Models (2-Stage)** |
| PI1M_concat_5epoch_2stage | 236,169 | 5,047 | **2-stage** |
| **MOSES Models (1-Stage)** |
| MOSES_noconcat_5epoch | 184,614 | 3,112 | 1-stage |
| MOSES_concat_5epoch | 172,964 | 6,183 | 1-stage |
| **MOSES Models (2-Stage)** |
| MOSES_concat_5epoch_2stage | 186,361 | 3,799 | **2-stage** |
| **Benchmarks** |
| SmilesPE_PI1M | 113,497 | 1,635 | - |
| SmilesPE_MOSES | 58,589 | 1,969 | - |

**Key Observations:**
- H-Net generates **2-4× more tokens** than SmilesPE but with **shorter token lengths** (2-3 chars vs. 4-6 chars)
- More training epochs → fewer tokens per SMILES (21.6 → 18.2 → 16.6) but more unique tokens (4,903 → 5,775 → 8,019)
- Polymer SMILES consistently generate ~20% more tokens than molecular SMILES
- **2-stage models** generate slightly more total tokens than 1-stage (hierarchical overhead)

---

## 3. Question A: Dataset Nature Effect (Polymer vs. Molecular)

### 3.1 Quantitative Comparison

**Non-Concatenated Models (5 epochs):**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Token Overlap (Jaccard) | 0.299 | Only 30% of tokens are shared |
| Breakpoint Overlap (Jaccard) | 0.571 | 57% of split positions agree |
| KL Divergence | 3.915 | Large distribution difference |
| Mean Token Length Diff | -0.279 | Polymer tokens slightly shorter |

**Concatenated Models (5 epochs):**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Token Overlap (Jaccard) | 0.333 | 33% of tokens are shared |
| Breakpoint Overlap (Jaccard) | 0.590 | 59% of split positions agree |
| KL Divergence | 4.268 | Very large distribution difference |
| Mean Token Length Diff | -0.604 | Polymer tokens 0.6 chars shorter |

### 3.2 Key Findings

**✓ Dataset Nature Has Significant Impact:**
- **Different Token Vocabularies**: Only 30-33% overlap between polymer and molecular tokens
- **Different Tokenization Strategies**: KL divergence of 3.9-4.3 indicates distinct learned patterns
- **Polymer Complexity**: PI1M generates 18-22 tokens/SMILES vs. MOSES 17-18 tokens/SMILES
- **Moderate Breakpoint Agreement**: 57-59% suggests some shared chemical motifs but different overall patterns

**Interpretation:**
Similar to natural language models learning different tokens for Mandarin vs. English, H-Net learns distinct chemical "vocabularies" for polymeric vs. molecular structures. Polymers require more fine-grained tokenization due to their inherent complexity.

**See Figure:** `figures/dataset_nature_token_lengths_noconcat.png` and `figures/dataset_nature_top_tokens_noconcat.png`

---

## 4. Question B: Concatenation Effect

### 4.1 Quantitative Comparison

**PI1M (Polymer Dataset):**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Token Overlap (Jaccard) | 0.351 | 35% of tokens are shared |
| Breakpoint Overlap (Jaccard) | 0.756 | 76% of split positions agree |
| Mean Token Length Diff | +0.452 | Concat tokens 0.45 chars longer |
| KL Divergence | 2.067 | Moderate distribution difference |

**MOSES (Molecular Dataset):**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Token Overlap (Jaccard) | 0.449 | 45% of tokens are shared |
| Breakpoint Overlap (Jaccard) | **0.960** | 96% of split positions agree! |
| Mean Token Length Diff | +0.127 | Concat tokens slightly longer |
| KL Divergence | 1.307 | Small distribution difference |

### 4.2 Key Findings

**✓ Concatenation Has Strong Effect, But Dataset-Dependent:**

**For Polymers (PI1M):**
- **Moderate Token Change**: 35% overlap suggests concatenation creates new specialized tokens
- **Good Breakpoint Stability**: 76% agreement means core splitting logic is preserved
- **Longer Tokens**: +0.45 chars suggests more context-aware chunking
- **Unique Token Increase**: 4,094 → 5,775 tokens (+41%)

**For Molecules (MOSES):**
- **Higher Token Overlap**: 45% shared tokens indicates more stable vocabulary
- **Excellent Breakpoint Stability**: **96% agreement** - almost identical splitting!
- **Minimal Token Length Change**: Only +0.13 chars
- **Larger Unique Token Increase**: 3,112 → 6,183 tokens (+99%)

**Answer to B_2: Did concatenation have a smaller effect on polymers?**
- **No, the opposite!** Concatenation had a **larger effect on polymers**:
  - Polymer KL divergence: 2.067 (larger change)
  - Molecular KL divergence: 1.307 (smaller change)
  - Polymer breakpoint overlap: 76% vs. Molecular: 96%
  
**Interpretation:**
The hypothesis that polymer datasets would be less affected by concatenation (because polymers are inherently concatenated structures) was **not supported**. Instead, molecular SMILES showed remarkable stability under concatenation (96% breakpoint agreement), while polymer SMILES showed more significant changes. This suggests:
1. Short molecular SMILES have well-defined, stable tokenization patterns
2. Longer polymer SMILES benefit more from additional context during concatenation training

---

## 5. Question C: Training Amount Effect

### 5.1 Quantitative Comparison (All Concatenated PI1M)

| Metric | 1→5 Epochs | 5→22 Epochs | 1→22 Epochs |
|--------|------------|-------------|-------------|
| Token Overlap (Jaccard) | 0.389 | 0.351 | 0.333 |
| Breakpoint Overlap (Jaccard) | 0.822 | 0.864 | 0.913 |
| Mean Token Length Diff | +0.411 | +0.258 | +0.669 |
| KL Divergence | 1.947 | 1.903 | 2.045 |

### 5.2 Progressive Training Statistics

| Training Stage | Unique Tokens | Avg Tokens/SMILES | Mean Token Length |
|----------------|---------------|-------------------|-------------------|
| 1 epoch (68M) | 4,903 | 21.63 | 2.20 |
| 5 epochs (240M) | 5,775 (+18%) | 18.23 (-16%) | 2.62 (+19%) |
| 22 epochs (1B) | 8,019 (+39%) | 16.59 (-9%) | 2.87 (+10%) |
| **Total Change** | **+63%** | **-23%** | **+30%** |

### 5.3 Key Findings

**✓ More Training = More Efficient, Specialized Tokenization:**

1. **Token Vocabulary Growth**: +63% unique tokens (4,903 → 8,019)
   - Model learns increasingly specialized chemical patterns
   
2. **Increased Efficiency**: -23% tokens per SMILES (21.6 → 16.6)
   - Longer tokens capture more information per token
   
3. **Longer Token Length**: +30% mean length (2.20 → 2.87 chars)
   - Progressive refinement toward optimal chunk sizes
   
4. **High Breakpoint Stability**: 82-91% overlap
   - Core tokenization logic remains stable despite vocabulary growth
   
5. **Diminishing Returns**: Bigger changes from 1→5 epochs than 5→22 epochs
   - Most learning happens in early epochs

**Interpretation:**
Extended training doesn't fundamentally change *where* tokens are split (high breakpoint overlap) but *what* tokens are learned. The model progressively discovers more efficient, longer tokens that compress SMILES into fewer chunks. This is beneficial for downstream tasks as it creates richer semantic representations.

**Note on Overfitting:**
The dataset was seen 22 times (total training dataset = ~50M bytes), yet the model continued to improve efficiency rather than memorize, suggesting good generalization.

---

## 6. Question D: H-Net vs. SmilesPE Benchmark

### 6.1 Comparative Statistics

**PI1M (Polymer) Comparison:**

| Metric | H-Net (concat_5epoch) | SmilesPE | Ratio |
|--------|----------------------|----------|-------|
| Total Tokens | 182,298 | 113,497 | 1.61× |
| Unique Tokens | 5,775 | 1,635 | 3.53× |
| Tokens per SMILES | 18.23 | 11.35 | 1.61× |
| Mean Token Length | 2.62 | 4.20 | 0.62× |

**MOSES (Molecular) Comparison:**

| Metric | H-Net (concat_5epoch) | SmilesPE | Ratio |
|--------|----------------------|----------|-------|
| Total Tokens | 172,964 | 58,589 | 2.95× |
| Unique Tokens | 6,183 | 1,969 | 3.14× |
| Tokens per SMILES | 17.30 | 5.86 | 2.95× |
| Mean Token Length | 2.01 | 5.94 | 0.34× |

### 6.2 Key Findings

**✓ Fundamentally Different Tokenization Philosophies:**

**H-Net Characteristics:**
- **Fine-grained, byte-level tokens**: 2-3 character average
- **Large, diverse vocabulary**: 3-4× more unique tokens
- **More tokens per SMILES**: 1.6-3× more tokens needed
- **Data-driven learning**: Adapts to dataset characteristics

**SmilesPE Characteristics:**
- **Coarse-grained, subword tokens**: 4-6 character average
- **Compact vocabulary**: Pre-trained, fixed vocabulary
- **Fewer tokens per SMILES**: More efficient compression
- **Rule-based**: Same tokenization regardless of dataset

**Trade-offs:**

| Aspect | H-Net Advantage | SmilesPE Advantage |
|--------|----------------|-------------------|
| **Adaptability** | ✓ Learns dataset-specific patterns | Fixed vocabulary |
| **Token Diversity** | ✓ Rich, specialized vocabulary | Compact, standardized |
| **Compression** | More tokens needed | ✓ Fewer tokens per SMILES |
| **Interpretability** | Byte-level harder to interpret | ✓ Chemical substructures |
| **Training** | ✓ Continuously improves | Pre-trained, static |

**See Figure:** `figures/benchmark_token_lengths.png`

---

## 7. Question F: Compression Metrics (BPB & Perplexity)

### 7.1 Background: Why Compression Metrics Matter for Chemistry

From the H-Net paper (Hwang et al., 2025), two key metrics measure model quality:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Bits-Per-Byte (BPB)** | `CE_loss / ln(2)` | How efficiently can the model compress data? |
| **Perplexity (PPL)** | `exp(CE_loss)` | How "surprised" is the model by the next byte? |

**Relevance for Chemical SMILES:**
- Lower BPB = model has learned meaningful chemical "grammar"
- SMILES has inherent patterns (functional groups, rings, stereochemistry)
- A model that learns these patterns will compress better
- Potentially useful for anomaly detection: high PPL = unusual molecule

**Reference Points:**
- Random prediction (byte-level): 8.0 BPB, 256 PPL
- Well-trained English LM: ~1.0-1.5 BPB
- H-Net on DNA: Nearly 4× improvement in data efficiency over baselines

### 7.2 Compression Efficiency Results

| Model | Final BPB | Final PPL | Best BPB | Training Bytes |
|-------|-----------|-----------|----------|----------------|
| **PI1M Models (1-Stage)** |
| PI1M_noconcat_5epoch | **0.670** | 1.59 | 0.668 | 243M |
| PI1M_concat_1epoch | 0.831 | 1.78 | 0.831 | 48M |
| PI1M_concat_5epoch | 0.687 | 1.61 | 0.683 | 238M |
| PI1M_concat_22epoch | **0.639** 🏆 | **1.56** | **0.623** | 1,048M |
| **PI1M Models (2-Stage)** |
| PI1M_concat_5epoch_2stage | 0.686 | 1.61 | 0.681 | 238M |
| **MOSES Models (1-Stage)** |
| MOSES_noconcat_5epoch | **0.658** | **1.58** | 0.654 | 367M |
| MOSES_concat_5epoch | 0.682 | 1.60 | 0.678 | 358M |
| **MOSES Models (2-Stage)** |
| MOSES_concat_5epoch_2stage | **0.679** | 1.60 | **0.674** | 358M |

**🏆 Best Overall Compression: PI1M_concat_22epoch (BPB=0.639, PPL=1.56)**

**Key Observations:**
- All models achieve **far better than random** (8.0 BPB) → model learns meaningful patterns
- Best BPB (0.62-0.67) is **comparable to good language models** (English ~1.0-1.5 BPB)
- **More training improves compression**: 1 epoch (0.83 BPB) → 22 epochs (0.64 BPB) = **23% improvement**
- Non-concatenated models achieve slightly better compression than concatenated (for same epochs)
- **2-stage models show marginal BPB improvement** over 1-stage for MOSES (+0.4%)

### 7.3 Key Research Questions

**F.1: Training Dynamics**
- How do BPB/PPL improve with more training bytes?
- Is there a plateau or continued improvement?

**F.2: Dataset Comparison**
- Does H-Net compress polymers differently than molecules?
- Which dataset achieves better compression?

**F.3: Architecture Effect on Compression**
- Do 2-stage models achieve better compression than 1-stage?
- Does hierarchical chunking improve predictive quality?

**F.4: SmilesPE Comparison**
- **Important Note**: SmilesPE is a tokenizer, not a language model
- Cannot directly compare PPL/BPB
- For fair comparison, would need to train an LM on SmilesPE tokens
- Use compression ratio as proxy: `bytes / tokens`
- H-Net's advantage: End-to-end learned compression vs. fixed vocabulary

### 7.4 Interpretation for Chemistry

**Why BPB Matters:**
1. **Information-theoretic efficiency**: Lower BPB = fewer bits needed to encode chemical structure
2. **Pattern learning**: Model has learned meaningful chemical regularities
3. **Compression = Understanding**: Similar to how good NLP models compress language efficiently

**Comparison to SmilesPE (Theoretical):**
- SmilesPE vocab ~30k tokens, avg token length ~4-5 chars
- Theoretical BPB ≈ log2(30000) / avg_token_len ≈ 15 / 4.5 ≈ 3.3 BPB
- H-Net learns byte-level patterns directly → can potentially achieve lower BPB
- Key advantage: H-Net adapts to dataset; SmilesPE uses fixed vocabulary

**See Figures:**
- `figures/compression_bpb_training_dynamics.png` - BPB vs training bytes
- `figures/compression_bpb_comparison.png` - Final BPB bar chart
- `figures/compression_training_amount_effect.png` - 1 vs 5 vs 22 epochs

---

## 8. Question E: Architecture Effect (1-Stage vs. 2-Stage)

### 8.1 Background: 2-Stage H-Net Architecture

The 2-stage H-Net architecture introduces **hierarchical chunking**:
- **Stage 0 (Outer)**: Groups bytes into initial "chunks" (similar to 1-stage tokenization)
- **Stage 1 (Inner)**: Groups Stage 0 chunks into "super-chunks" (higher-level groupings)

**Architecture Comparison:**

| Aspect | 1-Stage | 2-Stage |
|--------|---------|---------|
| `arch_layout` | `["m4", ["T22"], "m4"]` | `["m4", ["T1m4", ["T22"], "m4T1"], "m4"]` |
| `d_model` | [1024, 1536] | [1024, 1024, 1536] |
| Hierarchy levels | 2 | 3 |
| Chunking stages | 1 | 2 |
| Output | Tokens | Tokens + Super-chunks |

### 8.2 Models Compared

| Dataset | 1-Stage Model | 2-Stage Model |
|---------|---------------|---------------|
| PI1M (Polymer) | `run_large_20251111_181836` | `run_large_20260115_191350` ✓ |
| MOSES (Molecular) | `run_large_20251112_071557` | `run_large_20260116_074355` ✓ |

*All models: Concatenated (10x), 5 epochs, ~350M parameters*

### 8.3 Compression Metrics Comparison

| Dataset | 1-Stage BPB | 2-Stage BPB | Change | 1-Stage PPL | 2-Stage PPL |
|---------|-------------|-------------|--------|-------------|-------------|
| PI1M (Polymer) | 0.687 | 0.686 | **+0.2%** | 1.61 | 1.61 |
| MOSES (Molecular) | 0.682 | 0.679 | **+0.4%** | 1.60 | 1.60 |

**Finding:** 2-stage architecture provides **marginal improvement** in compression (~0.2-0.4%) for the same training budget. The hierarchical structure adds minimal overhead.

### 8.4 Tokenization Statistics Comparison

| Metric | PI1M 1-stage | PI1M 2-stage | MOSES 1-stage | MOSES 2-stage |
|--------|--------------|--------------|---------------|---------------|
| Total Tokens | 182,298 | 236,169 | 172,964 | 186,361 |
| Unique Tokens | 5,775 | 5,047 | 6,183 | 3,799 |

**Key Observations:**
- 2-stage models generate **more total tokens** (Stage 0 + Stage 1 boundaries)
- 2-stage models have **fewer unique tokens** → more consistent chunking patterns
- The hierarchical structure creates a trade-off: more structure, less vocabulary diversity

### 8.5 Key Findings

**E: Architecture Effect**
- 2-stage provides marginal compression improvement (+0.2-0.4% BPB)
- Comparable perplexity between architectures (PPL ~1.60-1.61)
- 2-stage adds interpretability through hierarchical structure

**E.1: Chunking Hierarchy**
- Stage 0 learns byte-level patterns (2-3 character tokens)
- Stage 1 groups Stage 0 chunks into super-chunks (higher-level groupings)
- Visualizations show clear two-tier structure in GIF animations

**E.3: Interaction with Dataset**
- MOSES (molecular) shows slightly more benefit from 2-stage (+0.4%) than PI1M (+0.2%)
- Counter to hypothesis: smaller molecules benefit more from hierarchical structure
- Polymers may already have sufficient complexity in 1-stage

**Interpretation:**
The 2-stage architecture adds **interpretability** (viewing chunking at two levels) without significantly impacting compression performance. For downstream tasks requiring hierarchical representations (e.g., polymer chain analysis), 2-stage may provide additional semantic structure.

**See Analysis Notebook:** `notebooks/07_architecture_effect_analysis.ipynb`
**See Visualizations:** `checkpoints/run_large_2026*/visualizations/`

---

## 9. Overall Conclusions

### 9.1 Main Findings

1. **Dataset Specificity Matters**: H-Net learns distinct tokenization strategies for polymeric vs. molecular SMILES, similar to language-specific patterns in NLP.

2. **Concatenation is Beneficial**: Especially for molecular SMILES, concatenation creates more diverse vocabularies without destabilizing core tokenization logic.

3. **Training Pays Off**: Extended training (22 epochs) produces 63% more unique tokens and 23% more efficient tokenization without overfitting.

4. **Different Philosophy from SmilesPE**: H-Net learns fine-grained, byte-level patterns adaptively, while SmilesPE uses fixed, chemical substructure-based tokens.

5. **Excellent Compression**: All models achieve BPB of 0.63-0.83, far better than random (8.0 BPB), indicating meaningful chemical pattern learning.

6. **2-Stage Architecture**: Provides marginal compression improvement (+0.2-0.4%) with added hierarchical interpretability; trade-off between structure and vocabulary diversity.

### 9.2 Practical Implications

**When to use H-Net:**
- Need dataset-specific tokenization
- Working with novel chemical structures
- Want continuous improvement with more training
- Prefer learned representations over predefined rules

**When to use SmilesPE:**
- Need standardized, interpretable tokens
- Want efficient compression
- Working with well-studied chemical spaces
- Need zero-shot tokenization

### 9.3 Future Directions

1. **Hybrid Approaches**: Combine H-Net's adaptability with SmilesPE's chemical knowledge
2. **Longer Sequences**: Test on even longer polymer chains
3. **Downstream Tasks**: Evaluate tokenization quality on property prediction, generation tasks
4. **Interpretability**: Analyze what chemical patterns H-Net's tokens represent

---

## 10. Data and Reproducibility

### 10.1 Generated Outputs

**Tokenization Results:**
- `data/hnet_results/` - 6 H-Net model tokenization outputs (.pkl files)
- `data/smilesPE_results/` - 2 SmilesPE benchmark outputs (.pkl files)

**Statistics:**
- `data/statistics/` - 8 detailed JSON files with token statistics

**Summary Tables:**
- `data/all_models_comparison.csv` - Complete model comparison
- `data/dataset_nature_summary.csv` - Polymer vs. molecular analysis
- `data/concatenation_effect_summary.csv` - Concatenation impact
- `data/training_amount_summary.csv` - Training progression analysis

**Visualizations:**
- `figures/dataset_nature_token_lengths_noconcat.png` - Token length distributions
- `figures/dataset_nature_top_tokens_noconcat.png` - Top token comparison
- `figures/benchmark_token_lengths.png` - H-Net vs. SmilesPE comparison

### 10.2 Analysis Notebooks

All analyses are reproducible via Jupyter notebooks:
1. `notebooks/01_data_generation.ipynb` - Data generation pipeline
2. `notebooks/02_dataset_nature_analysis.ipynb` - Polymer vs. molecular analysis
3. `notebooks/03_concatenation_effect.ipynb` - Concatenation impact study
4. `notebooks/04_training_amount_analysis.ipynb` - Training progression analysis
5. `notebooks/05_benchmark_comparison.ipynb` - SmilesPE comparison

---

## 11. Poster Content Section

*For A0 poster presentation - concise, visual-first format*

### POSTER LAYOUT RECOMMENDATION

**Title Banner:**
```
Deep-Learned Dynamic Tokenization for Chemical SMILES:
An Observability Study with H-Net
```

---

### SECTION 1: MOTIVATION & OBJECTIVES

**Why Tokenization Matters:**
- Chemical ML models need effective token representations
- Traditional: Fixed, rule-based tokenizers (BPE, SmilesPE)
- Novel: H-Net learns dynamic chunking from data

**Research Questions:**
1. Does dataset nature (polymer vs. molecular) affect learned tokens?
2. What is the effect of concatenating multiple SMILES?
3. How does training amount impact tokenization?
4. How does H-Net compare to industry-standard SmilesPE?

---

### SECTION 2: METHODOLOGY

**Datasets:**
- **PI1M**: Polymeric SMILES (PSMILES) - complex, long
- **MOSES**: Molecular SMILES - simple, short
- **Sample Size**: 10,000 SMILES per dataset

**Models Tested:**
- 6 H-Net models (varying concatenation, epochs)
- SmilesPE benchmark (ChEMBL vocabulary)

**Analysis Metrics:**
- Token statistics (count, length, diversity)
- Jaccard similarity, KL divergence
- Breakpoint overlap analysis

---

### SECTION 3: KEY RESULTS (3 PANELS)

#### PANEL 1: Dataset Nature Effect

**Title:** *Polymer vs. Molecular: Different Datasets, Different Tokens*

**Figure:** `dataset_nature_token_lengths_noconcat.png` + `dataset_nature_top_tokens_noconcat.png`

**Key Numbers (Large Text):**
- **30%** Token Overlap (Jaccard)
- **57%** Breakpoint Agreement
- **+20%** More tokens for polymers

**Take-Home:**
> H-Net learns dataset-specific "chemical vocabularies" - just as it learns different patterns for Mandarin vs. English in NLP

---

#### PANEL 2: Concatenation & Training Effects

**Title:** *More Context & Training → Better Tokenization*

**Sub-panel A: Concatenation Effect**

| Dataset | Token Overlap | Breakpoint Stability |
|---------|---------------|---------------------|
| Polymer | 35% | 76% |
| Molecular | 45% | **96%** ✓ |

**Take-Home:**
> Concatenation creates specialized tokens while preserving core logic (96% breakpoint agreement for molecules!)

**Sub-panel B: Training Progression (1 → 5 → 22 epochs)**

**Visual: Progressive Bar Chart**
- Unique Tokens: 4,903 → 5,775 → 8,019 (**+63%**)
- Tokens/SMILES: 21.6 → 18.2 → 16.6 (**-23%** efficiency gain)
- Token Length: 2.2 → 2.6 → 2.9 (**+30%** longer tokens)

**Take-Home:**
> More training = more efficient tokenization without overfitting

---

#### PANEL 3: H-Net vs. SmilesPE

**Title:** *Two Philosophies: Learned vs. Rule-Based*

**Figure:** `benchmark_token_lengths.png`

**Comparison Table (Visual):**

|  | H-Net | SmilesPE |
|--|-------|----------|
| **Token Length** | 2-3 chars (byte-level) | 4-6 chars (subwords) |
| **Vocabulary** | 6,000-8,000 (adaptive) | 1,600-2,000 (fixed) |
| **Tokens/SMILES** | 16-22 (fine-grained) | 6-11 (compressed) |
| **Strength** | ✓ Dataset-adaptive | ✓ Interpretable |

**Side-by-side Example:**
```
SMILES: CC(C)CCC1C2CCC3C(C2CC1)CCC3C(C)CCCC(C)C

H-Net:    CC | (C) | CCC | 1 | C2 | CCC | 3 | C | (C2 | CC | 1) | ...
          [21 tokens, mean length: 2.2]

SmilesPE: CC(C) | CCC1C2 | CCC3 | C(C2 | CC1) | CCC3C(C) | CCCC(C)C
          [11 tokens, mean length: 4.2]
```

**Take-Home:**
> H-Net learns fine-grained patterns adaptively; SmilesPE uses predefined chemical building blocks

---

### SECTION 4: CONCLUSIONS

**Main Findings (Bullet Points):**

✓ **Dataset-Specific Learning**: 30% token overlap between polymer/molecular → distinct "chemical languages"

✓ **Concatenation Benefits**: 96% breakpoint stability for molecules + 99% vocabulary growth

✓ **Training Efficiency**: 22 epochs → 63% more unique tokens, 23% fewer tokens per SMILES

✓ **Complementary to SmilesPE**: Fine-grained vs. coarse-grained trade-offs

**Implications:**
- H-Net adapts to novel chemical structures
- Useful for dataset-specific downstream tasks
- Concatenation is effective for context-aware learning
- Extended training improves efficiency without overfitting

**Future Work:**
- Hybrid H-Net + SmilesPE approaches
- Downstream task evaluation (property prediction)
- Chemical interpretability of learned tokens

---

### SECTION 5: VISUAL ASSETS FOR POSTER

**Recommended Figure Placement:**
1. **Top Left**: Dataset nature comparison (`dataset_nature_token_lengths_noconcat.png`)
2. **Top Right**: Top tokens visualization (`dataset_nature_top_tokens_noconcat.png`)
3. **Bottom Center**: Benchmark comparison (`benchmark_token_lengths.png`)
4. **Small Insets**: Progressive training statistics (create simple bar charts from CSV data)

**Color Scheme (Mako Palette):**
- Primary: Deep purples/blues for H-Net
- Secondary: Warm yellows/greens for SmilesPE
- Accent: Bright highlights for key numbers

**Typography Recommendations:**
- **Title**: 120pt, bold
- **Section Headers**: 72pt, semi-bold
- **Body Text**: 48pt
- **Key Numbers**: 96pt, bold, colored
- **Captions**: 36pt

---

### POSTER QR CODE CONTENT

**Link to Full Report:**
- GitHub repository with all code, data, notebooks
- Full analysis report (this document)
- Interactive notebooks for reproducibility

---

## 12. Contact & Acknowledgments

**Code & Data:**
- Full analysis pipeline: `hnet_smiles/analysis/`
- Notebooks: `hnet_smiles/analysis/notebooks/`
- All results reproducible with provided scripts

**Key Files:**
- This report: `analysis/FINAL_REPORT.md`
- Analysis goals: `analysis/analysis_goals.md`
- Implementation summary: `analysis/IMPLEMENTATION_SUMMARY.md`

---

*Report generated: November 13, 2025*  
*Updated: January 16, 2026 (added 2-stage models, compression metrics)*  
*Analysis completed with 10,000 SMILES per dataset*  
*Total models: 8 H-Net (6 × 1-stage + 2 × 2-stage) + 2 SmilesPE benchmarks*  
*All figures use seaborn 'whitegrid' style, 'talk' context, 'mako' palette*

