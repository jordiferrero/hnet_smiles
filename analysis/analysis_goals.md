# Document setting up an analysis strategy and key questions to answer
# Goal: Observability study on tokensiation of SMILES chemical strings using H-Net deep-learnt dynamic chunking/tokenisation

## Datasets ran

### 1-Stage Architecture Models
These are all the trained models using the **1-stage H-Net architecture** (`arch_layout: ["m4", ["T22"], "m4"]`):
- Large PI1M no concatenation, 5 epoch (240M bytes) `hnet_smiles/checkpoints/run_large_20251111_075600`
- Large PI1M with 10-PSMILES concatenation, 5 epoch (240M bytes) `hnet_smiles/checkpoints/run_large_20251111_181836`
- Large PI1M with 10-PSMILES concatenation, 22 epoch (1B bytes) `hnet_smiles/checkpoints/run_large_20251112_150502`
- Large PI1M with 10-PSMILES concatenation, 1 epoch (68M bytes) `hnet_smiles/checkpoints/run_large_20251113_181705`
- Large MOSES no concatenation, 5 epoch (360M bytes) `hnet_smiles/checkpoints/run_large_20251113_074900`
- Large MOSES with 10-SMILES concatenation, 5 epoch (360M bytes) `hnet_smiles/checkpoints/run_large_20251112_071557`

### 2-Stage Architecture Models
These are the trained models using the **2-stage H-Net architecture** (`arch_layout: ["m4", ["T1m4", ["T22"], "m4T1"], "m4"]`):
- Large PI1M with 10-PSMILES concatenation, 5 epoch (240M bytes), 2-stage `hnet_smiles/checkpoints/run_large_2stage_PI1M` *(pending)*
- Large MOSES with 10-SMILES concatenation, 5 epoch (360M bytes), 2-stage `hnet_smiles/checkpoints/run_large_2stage_MOSES` *(pending)*

The 2-stage architecture introduces a **hierarchical chunking mechanism** where:
- **Stage 0 (Outer)**: Chunks bytes into initial tokens (byte-level → chunk-level)
- **Stage 1 (Inner)**: Groups Stage 0 chunks into "super-chunks" (chunk-level → super-chunk-level)

The goal was to compare the effect that following parameters have on how "dynamic chunking" learns to chunk and create tokens:
- The nature of the dataset: polymeric (longer PSMILES from PI1M) vs. molecular (shorter SMILES from MOSES). Does dynamic tokenisation learn different chemical interpretations/tokens depending on the nature of the dataset (in natural language, the same model learnt different and richer tokens for Mandarin vs. English). Polymers are inherently more complex/different than molecules/monomers, but not drastically.
- Effect of concatenation: because the H-Net architecture was designed to work for natural language (longer texts), concatenation of multiple (10) SMILES stings may result in better learning because the learnt vectors have more context? However it may add noise to later concatenated strings from independent SMILES before (aka concatenation is not a physical phenomenon here). " " character was used for concatenation, as it has no meaning in SMILES and PSMILES. Also, did concatenation have a smaller effect in the change of learnt tokens for the polymer dataset than for the molecular dataset (because polymers are inherently chemical concatenations of monomers, so the polymer dataset is inherently concatenated by nature)?
- The amount of training data used: does training on 1B bytes significantly change the tokenisation strategy, compared to ~250M bytes. Note the total available training datasets are only ~50/80M bytes (polymer and molecular, respectively), so the same dataset is seen several times (some overfitting may happen). Most runs were don at 5 epoch.

### Experimental Matrix

**1-Stage Architecture:**

|                       | PSMILES (polymer) | SMILES (molecular) |
|-----------------------|-------------------|---------------------|
| No concatenation      | 5 epoch           | 5 epoch             |
| Concatenation         | 1, 5 and 22 epoch | 5 epoch             |

**2-Stage Architecture:**

|                       | PSMILES (polymer) | SMILES (molecular) |
|-----------------------|-------------------|---------------------|
| No concatenation      | -                 | -                   |
| Concatenation         | 5 epoch           | 5 epoch             |

**Architecture Comparison (matched conditions - 5 epoch, concatenated):**

| Architecture | PSMILES (polymer) | SMILES (molecular) |
|--------------|-------------------|---------------------|
| 1-Stage      | ✓                 | ✓                  |
| 2-Stage      | ✓ *(pending)*     | ✓ *(pending)*      |


## Data analysis
### Analysis data generation
For each of the 6 trained models above:
1. Load the `hnet_smiles/checkpoints/run_large_{timestamp}/checkpoints/checkpoint_bytes_best.pt` model and run token chunking inference for each of the entries in the full dataset (either `PI1M` or `MOSES` depending on `hnet_smiles/checkpoints/run_large_{timestamp}/metadaat.json`). This should be stored in a memory-effective way to then be able to do statistics and analysis on the tokens/chunks for the whole dataset effectively.
2. Generate some statistics and analysis on the tokens/chunks and store in memory as datasets and figures/visualisations, such as:
- What are the top 50 most frequent token sequences learnt in the whole dataset using this trained model?
- What are the top 50 most frequent break point characters during chunking learnt in the whole dataset using this trained model?
- Any other interesting metrics (median length of tokens)...
3. Compare the token sequences learnt in the whole dataset using each trained model to the tokens generated for the same dataset by the industry-standard tokeniser SmilesPE (SMILES Pair Encoding). SmilesPE is a Python package on PyPI (simple pip install SmilesPE) with a small, self‑contained API and a ready‑made vocabulary you can use immediately. It works as such:
```python
import codecs
from SmilesPE.tokenizer import SPE_Tokenizer

# download SPE_ChEMBL.txt from the SmilesPE repo (pretrained vocab)
spe = SPE_Tokenizer(codecs.open('SPE_ChEMBL.txt', 'r', 'utf-8'))

smiles  = 'c1ccccc1O'           # phenol (SMILES)
psmiles = '[*]CC(=O)OCC[*]'     # simple polyester repeat unit (PSMILES)

# tokenize -> space-separated string of substrings
print(spe.tokenize(smiles))      # e.g. "c1ccccc1 O"
print(spe.tokenize(psmiles))     # e.g. "[*] CC(=O) OCC [*]"

# if you prefer a list:
tokens = spe.tokenize(psmiles).split(' ')
```
- Run this benchmark tokeniser to the same dataset (`PI1M` or `MOSES`) first. Store these benchmark results independently for each dataset. Run once.
- Then run statistical comparisons for each of the token distributions from each of the trained models against this benchmark SmilesPE tokeniser. Calculate the quantitative differences between them (e.g. difference in mean token length, similer top tokens...)



### Analysis evaluation
Based on all the different produced results, datasets and figures for the 6 models, attempt to answer the following questions using quantitative and visualisation answers. Make sure to only compare apples with apples (e.g. do not compare a concatenated-trained model with a non-concatenated model OR a 5-epoch model vs. a 22-epoch model, unless concatenation or training-size is the question of study, respectively; only compare two models that were ran under same conditions except the one variable parameter). Here are some of the quesions:
a. How does the nature of the dataset - polymeric (longer PSMILES from PI1M) vs. molecular (shorter SMILES from MOSES) - affect the dynamic tokenisation learnt? Compare the tokens/breakpoint-characters for non-concatenated polymer vs. molecular 5-epoch runs (or the concateded equivalent runs). Do we see different chemical interpretations/tokens depending on the nature of the dataset (in natural language, the same model learnt different and richer tokens for Mandarin vs. English). Polymers are inherently more complex/different than molecules/monomers, but not drastically. 
b_1. What is the effect of concatenation of SMILES strings on the tokens learnt? Because the H-Net architecture was designed to work for natural language (longer texts), concatenation of multiple (10) SMILES stings may result in better learning because the learnt vectors have more context? However it may add noise to later concatenated strings from independent SMILES before (aka concatenation is not a physical phenomenon here). " " character was used for concatenation, as it has no meaning in SMILES and PSMILES. 
b_2. Did concatenation have a smaller effect in the change of learnt tokens for the polymer dataset than for the molecular dataset (because polymers are inherently chemical concatenations of monomers, so the polymer dataset is inherently concatenated by nature)?
c. How did the amount of training data used affect the tokens learnt? This only has been studied for the polymer dataset. Does training on 1B bytes (22 epoch) significantly change the tokenisation strategy, compared to ~250M bytes (5 epoch), or just 68M bytes (1 epoch). You can compare here: "Large PI1M with 10-PSMILES concatenation, 1 epoch (68M bytes) `hnet_smiles/checkpoints/run_large_20251113_181705`" vs. "Large PI1M with 10-PSMILES concatenation, 5 epoch (240M bytes) `hnet_smiles/checkpoints/run_large_20251111_181836`" vs. "Large PI1M with 10-PSMILES concatenation, 22 epoch (1B bytes) `hnet_smiles/checkpoints/run_large_20251112_150502`". This provides a clean comparison across 1, 5, and 22 epochs all with the same concatenation strategy. Note the total available training dataset is only ~50M bytes (1 epoch for polymer), so the same dataset is seen several times (some overfitting may happen).
d. How does this deep-learnt dynamic tokeniser results in different tokens compared to the benchmark using SmilesPE tokenizer, based on the comparison data.

### 2-Stage Architecture Analysis (Section E)
These questions investigate the effect of using the 2-stage hierarchical H-Net architecture vs. the 1-stage architecture:

e. **Architecture Effect (1-stage vs. 2-stage)**: Compare the tokenization of 1-stage vs. 2-stage models under identical conditions (same dataset, concatenation, epochs). For a fair comparison, use:
   - PI1M: 1-stage (`run_large_20251111_181836`) vs. 2-stage (`run_large_2stage_PI1M`)
   - MOSES: 1-stage (`run_large_20251112_071557`) vs. 2-stage (`run_large_2stage_MOSES`)
   
   Metrics to compare:
   - Token overlap (Jaccard similarity of token vocabularies)
   - Breakpoint character distributions
   - Token efficiency (tokens per SMILES)
   - Mean token length distributions
   - KL divergence of token frequency distributions

e.1. **Chunking Hierarchy**: For 2-stage models, analyze how the two chunking levels relate:
   - What is the typical Stage 0 chunk size (in bytes)?
   - What is the typical Stage 1 super-chunk size (in Stage 0 chunks)?
   - Do the stages learn different patterns? (e.g., Stage 0 = atoms/bonds, Stage 1 = functional groups)
   - Visualize nesting patterns: how many Stage 0 chunks per Stage 1 super-chunk?

e.2. **Chemistry Interpretation**: What chemical patterns does each chunking stage learn?
   - Stage 0: Do boundaries occur at atomic/bond level?
   - Stage 1: Do super-chunk boundaries align with functional group boundaries?
   - Compare top tokens/breakpoints between stages
   - Are Stage 1 super-chunks more chemically meaningful than Stage 0 tokens?

e.3. **Interaction with Dataset**: Does the 2-stage architecture benefit polymers more than molecules?
   - Compare the "improvement" (or change) from 1-stage to 2-stage for PI1M vs. MOSES
   - Hypothesis: Polymers have more inherent hierarchy (monomers → chains → polymers), so 2-stage may learn this structure better
   - Metrics: Compare token efficiency gains, vocabulary diversity changes

e.4. **Interaction with Concatenation**: Does 2-stage + concatenation show synergy?
   - Since we only have 2-stage with concatenation, compare:
     - 1-stage no-concat vs. 1-stage concat (existing analysis)
     - 1-stage concat vs. 2-stage concat (architecture effect with concat)
   - Does 2-stage amplify or reduce the concatenation effect?

### Compression Metrics Analysis (Section F)
These questions investigate the compression efficiency and predictive quality using BPB and Perplexity metrics (inspired by the H-Net paper):

f. **Overall Compression Efficiency**: What is the final validation BPB for each H-Net model?
   - BPB = CE_loss / ln(2) - measures bits needed per byte
   - PPL = exp(CE_loss) - measures "surprise" per token
   - Lower values = better compression/prediction
   - Theoretical max BPB: 8.0 (random prediction for byte-level)
   
f.1. **Training Dynamics**: How do BPB/PPL improve with more training bytes?
   - Plot BPB vs training bytes for all models
   - Is there a plateau or continued improvement?
   - Compare 1 vs 5 vs 22 epochs

f.2. **Dataset Comparison**: Does H-Net compress polymers differently than molecules?
   - Compare final BPB between PI1M and MOSES models
   - Hypothesis: Different chemical structures may have different compressibility

f.3. **Architecture Effect on Compression**: Do 2-stage models achieve better compression?
   - Compare 1-stage vs 2-stage BPB under identical conditions
   - Does hierarchical chunking improve predictive quality?

f.4. **Concatenation Effect on Compression**: Does concatenation improve BPB?
   - Compare concatenated vs non-concatenated models

f.5. **SmilesPE Comparison**:
   - **Important**: SmilesPE is a tokenizer, not a language model
   - We cannot directly compute PPL/BPB for SmilesPE
   - Use compression ratio as proxy: bytes / tokens
   - For fair comparison, would need to train an LM on SmilesPE tokens
   - H-Net's advantage: End-to-end learned compression vs. fixed vocabulary

**Why These Metrics Matter for Chemistry:**
- BPB measures how efficiently H-Net learns chemical "grammar"
- Lower BPB = model has learned meaningful chemical patterns (functional groups, rings, etc.)
- Can compare to NLP benchmarks: English text ~1.0-1.5 BPB, DNA with H-Net ~4× improvement
- Potentially useful for anomaly detection: high PPL = unusual molecule

## Reporting
For all the analysis scripts, processed data and visualisations, keep everything in the `hnet_smiles/analysis` directory.
Please use Python and seaborn `whitegrid` style, `talk` context. Use seaborn `mako` color palette. Use matplotlib for most plots, or seaborn if necessary. Some of these figures I will use for a poster presentation (A0). I like to have jupyter notebooks for data analysis scripts where possible.
Create a final report with all the final results.