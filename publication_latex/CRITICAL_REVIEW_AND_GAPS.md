# Critical Review & Gaps Analysis for ICML 2026 Submission

## Overall Assessment

**Strengths of Current Work:**
- ✅ Novel angle: First systematic study of dynamic tokenization for chemistry
- ✅ Comprehensive analysis: Multiple variables systematically tested (dataset, concatenation, epochs, architecture)
- ✅ Quantitative rigor: Good metrics suite (Jaccard, KL divergence, BPB, breakpoint analysis)
- ✅ Practical validation: Property prediction demonstrates real-world utility
- ✅ Clear narrative: Dataset-specificity finding is compelling and novel

**Current Weaknesses (Honest Assessment):**
- ⚠️ Limited downstream validation (only 4 property prediction tasks)
- ⚠️ Missing comparison with other learned tokenizers (BLT, CANINE)
- ⚠️ No interpretability analysis of what chemical patterns tokens represent
- ⚠️ Single model size (350M) – no scaling analysis
- ⚠️ Property prediction section is thin for supporting "foundation model" claims

---

## 🔴 CRITICAL GAPS (Must Address Before Submission)

### Gap 1: Weak Baseline Comparisons for Tokenization

**Issue:** Only SmilesPE is compared. Reviewers will ask: "Why not compare to other learned tokenization methods?"

**Required Additions:**
1. **BPE trained on SMILES**: Train a standard BPE tokenizer on PI1M and MOSES, compare vocabulary
2. **Character-level baseline**: Simple character tokenization metrics
3. **Atom-level tokenizer**: Use RDKit to extract atom-level tokens (more chemistry-aware)

**Effort:** Medium (~1-2 days)

**Suggested Experiment:**
```
| Tokenizer | Unique Tokens | Tokens/SMILES | Downstream Tg MAE |
|-----------|---------------|---------------|-------------------|
| Character | 50-70 | ~50 | ? |
| Atom-level | ~100 | ~20 | ? |
| BPE (trained) | ~8K | ~15 | ? |
| SmilesPE | ~2K | ~11 | 24.8°C |
| H-Net | ~6K | ~17 | 26.6°C |
```

---

### Gap 2: Missing Interpretability Analysis

**Issue:** "What chemical patterns do H-Net tokens represent?" is never answered. This is a major gap for a chemistry paper at ICML.

**Required Analysis:**
1. **Token-to-structure mapping**: Cluster top-100 tokens by chemical meaning
   - Are tokens aligned with functional groups? (–OH, –COOH, rings, etc.)
   - Do tokens respect atom boundaries or split atoms?
2. **Attention visualization**: What does the boundary predictor attend to?
3. **Chemical validity analysis**: Do H-Net tokens preserve chemical syntax?

**Suggested Experiment:**
```python
# For top 50 H-Net tokens, manually annotate:
# - Chemical interpretation (e.g., "CC" = ethyl, "(=O)" = carbonyl)
# - Alignment with functional groups
# - Create a figure showing token → chemical structure mapping
```

**Effort:** Medium-High (~2-3 days for manual annotation + visualization)

---

### Gap 3: Limited Property Prediction Tasks

**Issue:** Only 4 tasks (2 polymer, 2 molecular) is too few to claim "foundation model" capabilities.

**Required Additions:**
1. **More MoleculeNet tasks**: Add at least 3-4 more:
   - Solubility (ESOL)
   - Toxicity (Tox21, ClinTox)
   - HIV activity
   - SIDER (side effects)
2. **Larger polymer dataset**: Current PI1M subset is 10K samples; show scaling
3. **Multi-task learning**: Can one H-Net model transfer across tasks?

**Effort:** Medium (~2-3 days for additional experiments)

---

### Gap 4: No Statistical Significance Testing

**Issue:** Many comparisons (e.g., "H-Net 0.950 AUC vs RDKit 0.927") lack statistical tests.

**Required:**
1. Report confidence intervals (already have std, good)
2. Paired t-tests or Wilcoxon tests for key comparisons
3. Bootstrap confidence intervals for Jaccard/KL metrics

**Effort:** Low (~1 day)

---

### Gap 5: Missing Ablation Studies

**Issue:** No ablation of H-Net architecture components. Why this specific configuration?

**Suggested Ablations:**
1. **Load balancing loss weight**: 0.01 vs. 0.001 vs. 0.1
2. **Hierarchical LR modulation**: With vs. without
3. **Number of Mamba blocks**: m2 vs. m4 vs. m8
4. **Boundary prediction mechanism**: Learned vs. fixed-length chunks

**Effort:** High (~1 week for training ablation models)

**Recommendation:** At minimum, include 2-3 key ablations (LB loss, LR modulation)

---

## 🟡 IMPORTANT GAPS (Strongly Recommended)

### Gap 6: No Generative Task Validation

**Issue:** Dynamic tokenization should improve generation quality. Currently only evaluate compression and classification.

**Suggested Experiments:**
1. **Molecule generation**: Use H-Net as tokenizer for autoregressive generation
   - Validity rate (RDKit parseable)
   - Novelty (not in training set)
   - Diversity (internal diversity)
2. **Reconstruct SMILES**: Encode → decode accuracy

**Effort:** Medium-High (~3-4 days)

---

### Gap 7: Compute Cost Analysis Missing

**Issue:** No discussion of training time, inference speed, memory usage. Reviewers will ask about practicality.

**Required:**
1. Training time per model (already have ~1-3 days)
2. Inference throughput: SMILES/second for tokenization
3. Memory footprint comparison: H-Net vs. SmilesPE
4. GPU requirements

**Format:**
```
| Aspect | H-Net | SmilesPE |
|--------|-------|----------|
| Training | 1-3 days (GPU) | None (pretrained) |
| Inference | ? SMILES/sec | ? SMILES/sec |
| Memory | ? GB | < 1 GB |
```

**Effort:** Low (~0.5 days)

---

### Gap 8: Cross-Domain Transfer Not Tested

**Issue:** Claim dataset-specificity is important, but don't test: Can polymer-trained H-Net tokenize molecules, and vice versa?

**Suggested Experiment:**
```
| Train Dataset | Eval Dataset | Tokens/SMILES | Downstream MAE |
|---------------|--------------|---------------|----------------|
| PI1M | PI1M | 18.2 | 26.6°C (Tg) |
| PI1M | MOSES | ? | ? (BBBP) |
| MOSES | MOSES | 17.3 | 0.95 (BBBP) |
| MOSES | PI1M | ? | ? (Tg) |
```

**Effort:** Low (~1 day using existing models)

---

### Gap 9: 2-Stage Architecture Underexplored

**Issue:** 2-stage model shows minimal improvement (+0.2-0.4% BPB). Why include it? Need stronger justification or deeper analysis.

**Required:**
1. What do Stage 0 vs. Stage 1 chunks represent?
2. Visualization of hierarchical chunking patterns
3. Use case justification: When is 2-stage beneficial?

**Alternative:** Consider removing 2-stage from main paper if no clear benefit; mention in future work.

---

### Gap 10: Missing Error Analysis

**Issue:** When does H-Net fail? No analysis of failure modes.

**Suggested Analysis:**
1. **High-PPL molecules**: Which SMILES have high perplexity? Are they unusual/valid?
2. **Property prediction errors**: What structural features cause large prediction errors?
3. **Tokenization failures**: Any SMILES that H-Net tokenizes poorly?

**Effort:** Low-Medium (~1-2 days)

---

## 🟢 NICE-TO-HAVE (Time Permitting)

### Gap 11: No Comparison with Other Chemical LMs

- Compare H-Net embeddings with ChemBERTa, MolBERT, SMI-TED embeddings
- Would strengthen "foundation model" claim
- **Effort:** Medium (~3 days)

### Gap 12: No Pre-training on Larger Chemical Corpora

- Current training on ~1M molecules is small by LLM standards
- Pre-training on ChEMBL (2M), PubChem (100M+) would be more compelling
- **Effort:** High (~1-2 weeks)

### Gap 13: Multi-modal Extension

- Combine SMILES tokenization with 3D coordinates
- **Effort:** High (out of scope for this paper)

---

## Reviewer Anticipation: Likely Questions & Responses

### Q1: "Why use byte-level instead of atom-level tokenization?"
**Current answer:** Flexibility, no domain knowledge required
**Stronger answer needed:** Empirical comparison showing byte-level + H-Net > atom-level

### Q2: "How does this compare to BPE trained on chemical data?"
**Current answer:** Not addressed
**Action:** Add BPE baseline (Gap 1)

### Q3: "What chemical patterns do the tokens represent?"
**Current answer:** Not addressed
**Action:** Add interpretability analysis (Gap 2)

### Q4: "Is the 2.3% BBBP improvement significant?"
**Current answer:** Report std only
**Action:** Add statistical significance tests (Gap 4)

### Q5: "Why only 4 property prediction tasks?"
**Current answer:** Not justified
**Action:** Add more tasks (Gap 3)

### Q6: "What's the computational overhead?"
**Current answer:** Not addressed
**Action:** Add compute analysis (Gap 7)

---

## Priority-Ordered Action Items

### MUST DO (Before Submission)
1. ⭐ Add BPE baseline comparison (Gap 1) — **Effort: 2 days**
2. ⭐ Add statistical significance tests (Gap 4) — **Effort: 1 day**
3. ⭐ Add compute cost analysis (Gap 7) — **Effort: 0.5 days**
4. ⭐ Cross-domain transfer experiment (Gap 8) — **Effort: 1 day**

### SHOULD DO (Significantly Strengthens Paper)
5. Add 2-3 more MoleculeNet tasks (Gap 3) — **Effort: 2 days**
6. Basic interpretability analysis (Gap 2) — **Effort: 2 days**
7. Error analysis section (Gap 10) — **Effort: 1.5 days**

### NICE TO HAVE (If Time Permits)
8. Generative task validation (Gap 6)
9. Ablation studies (Gap 5)
10. ChemBERTa comparison (Gap 11)

---

## Recommended Paper Framing Adjustments

### Current Framing Issues:
1. **"Foundation model"** claim is too strong with only 4 tasks
2. **Property prediction** takes too much space (1/3 of paper) for supporting evidence
3. **2-stage architecture** is underexplored but takes significant space

### Recommended Adjustments:
1. **Reframe** as "Dynamic tokenization study" with "preliminary foundation model evidence"
2. **Reduce** property prediction to 0.5-0.75 pages (proof-of-concept, not main contribution)
3. **Either** strengthen 2-stage analysis **or** move to appendix
4. **Add** interpretability section (even if brief) to main paper
5. **Add** computational cost comparison

---

## Estimated Timeline to Address Gaps

| Priority | Days Needed | Gap |
|----------|-------------|-----|
| MUST | 2 | BPE baseline |
| MUST | 1 | Statistical tests |
| MUST | 0.5 | Compute analysis |
| MUST | 1 | Cross-domain transfer |
| SHOULD | 2 | More MoleculeNet tasks |
| SHOULD | 2 | Interpretability |
| SHOULD | 1.5 | Error analysis |
| **TOTAL** | **~10 days** | |

---

## Final Verdict

**Current acceptance probability:** 40-50% (novel idea, but significant gaps)

**With MUST-DO gaps addressed:** 60-70% (solid observability study)

**With SHOULD-DO gaps addressed:** 75-85% (strong ICML paper)

**Key selling points to emphasize:**
1. **Novelty**: First systematic study of learned tokenization for chemistry
2. **Insight**: Dataset-specificity matters (30% overlap is striking)
3. **Practical**: Embeddings work for property prediction
4. **Reproducibility**: Clear experimental setup, all configurations documented

**Potential fatal flaw:** If reviewers view this as "applying existing method (H-Net) to new domain (chemistry)" without sufficient novelty, paper may be rejected. Counter by emphasizing:
- Unique insights about chemical tokenization behavior
- Systematic analysis with clear recommendations
- Foundation for future chemical LM work

---

*Critical review prepared: January 2026*
*Recommendation: Address MUST-DO gaps before submission*


