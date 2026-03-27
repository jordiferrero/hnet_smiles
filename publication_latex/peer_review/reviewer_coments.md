Official Review of Submission15667 by Reviewer iQmP
Official Reviewby Reviewer iQmP17 Mar 2026, 14:14 (modified: 24 Mar 2026, 14:30)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer iQmPRevisions
Summary:
This paper presents an empirical study exploring the application of Hierarchical Networks (H-Net) — an architecture originally designed for dynamic tokenization in NLP — to chemical SMILES strings. By training 8 models (each with 350M parameters) on two chemical datasets (PI1M for polymers and MOSES for small molecules), the authors evaluate how dataset characteristics, training strategies, and hierarchical architecture variants affect tokenization behavior. The study reports three main findings: (1) H-Net learns domain-specific "chemical vocabularies," with only 30% token overlap between the polymer and molecule datasets; (2) performance improves with longer training, with compression efficiency (BPB) improving to 0.64; (3) frozen H-Net embeddings outperform traditional RDKit descriptors on several classification benchmarks (e.g., BBBP AUC 0.95 vs. 0.93) but lag behind on regression tasks.

Strengths And Weaknesses:
Strengths:

The authors employ a well-structured empirical study that isolates the effects of dataset properties, concatenation strategies, and architecture depth, providing a framework for evaluating tokenization behavior.
The quantitative evidence that H-Net develops distinct vocabularies for different chemical domains (30% overlap) is a compelling finding. This suggests that dynamic tokenization can successfully adapt to the unique structural motifs of polymers versus small molecules.
The observation that the model respects atom boundaries approximately 70% of the time without explicit chemical supervision suggests that byte-level prediction naturally captures chemical structure.
Weaknesses:

The lack of comparison with established chemical language models (e.g., ChemBERTa, MolBERT) is a significant omission. Comparing embeddings solely against handcrafted RDKit features is insufficient to demonstrate H-Net's utility and state-of-the-art status for property prediction.
The reported finding that approximately 60% of unique tokens appear only once suggests the model may be memorizing noise rather than learning generalizable chemical patterns. The paper lacks rigorous ablation studies to determine whether these rare tokens contribute to downstream performance.
This work is an application study of an unmodified NLP architecture, H-Net (Hwang et al., 2025). There is a lack of chemistry-aware modifications.
H-Net underperforms RDKit on all five regression tasks (Tg, MAC, Lipophilicity, ESOL, FreeSolv), sometimes by a large margin. The explanation that "H-Net captures high-level structural patterns relevant to classification" but "misses fine-grained quantitative features" is speculative.
The scaling law analysis fits a power law on only three data points (68M, 340M, 1.05B bytes), which is insufficient to claim robust scaling trends.
Soundness: 2: fair
Presentation: 3: good
Significance: 2: fair
Originality: 2: fair
Key Questions For Authors:
Given that 60% of tokens are hapax legomena, can you report downstream model performance when these rare tokens are pruned or capped by a frequency threshold?
Can you provide 95% confidence intervals and p-values for the property prediction results in Table 6 to confirm that the reported improvements over RDKit are statistically robust across cross-validation folds?
Limitations:
Yes. The study is limited by small-scale/narrow-domain datasets, the lack of comparison with modern chemical language models, and potential overfitting to rare tokens.

Overall Recommendation: 3: Weak reject: A paper with clear merits, but also some weaknesses, which overall outweigh the merits. Papers in this category require revisions before they can be meaningfully built upon by others. Please use sparingly.
Confidence: 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.

---

Official Review of Submission15667 by Reviewer u1dD
Official Reviewby Reviewer u1dD12 Mar 2026, 15:23 (modified: 24 Mar 2026, 14:30)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer u1dDRevisions
Summary:
This work aims to evaluate whether dynamic tokenization is better than classical static tokenizers used for the chemical sciences. For this, they simply use the same H-net architecture as in the original paper, and train it using different strategies, such as concatenating or not the SMILES, using different SMILES, or using a two-stage hierarchical training (two levels of chunking).

With this, they then compare the different models that they trained, to see the effects of the tokenization, how the tokenization scales, and some analysis of the tokens that are being constructed.

At the end, to prove and show the effect of the application of dynamic tokenization for chemical applications, they apply this tokenization to some property prediction cases for molecules (mostly using MoleculeNet) and polymers predicting Tg.

Strengths And Weaknesses:
Strengths
I think the problem they target is really nice. At the end, how tokenization impacts the modelling in chemistry, and if a more chemically informed tokenization can really help, is an unresolved problem that really has a lot of importance, even affecting LLMs nowadays.
Weaknesses
Their history and their experiments are a bit disconnected.
Even the history itself reads differently in the different sections of the work
In the abstract, the claim is about simply investigating how dynamic tokenization works in chemistry, or basically training H-Nets for chemistry, which, honestly, I believe has 0 impact.
However, during the introduction, they claim they aim to investigate how the different tokenization would affect the different fields of chemistry, and they explicitly mention and work with the polymer example, which I believe is a really interesting case
Polymers use PSMILES, which are traditional SMILES but use the “\*” symbol as an indicator for where the repeating unit is connected.
To me, until this point, I was assuming that obviously the tokenization is going to be different, as those added symbols are unique in PSMILES and will change the structures completely, and I was expecting a comparison of whether such changes have a great impact in modelling, and how dynamic tokenization could avoid having different rigid tokenizers for these two different cases
However, the authors simply compared the different tokens produced when training H-nets only on SMILES and only on PSMILES.
I am sorry, but I think this adds nothing to the scientific community.
Like, what do I win by knowing that more training compute produces more unique tokens, or some of the other conclusions like the token length distribution between both representations (fig. 3, where both distributions look very similar)?
The only comparison that they have with the rigid tokenizers is Table 5, where they compare the token length and the tokens per SMILES. What does this tell people about how to train a model?
Section 5, where using the produced tokens as input for property prediction problems, the results in regression problems are still better using RDKit features; and in classification problems, where the dynamic tokenization performs better, the improvements are minimal, and since the table lacks error bars, I would say probably the results are very comparable.
And if one takes into account the bigger computational overhead that the H-nets carry, I would say it is not worth it.
Soundness: 3: good
Presentation: 3: good
Significance: 1: poor
Originality: 1: poor
Key Questions For Authors:
What is the main message that you want to share with your work?
What is the real effect of dynamic tokenization compared with rigid tokenization?
Is it surprising that there is overlap between the tokens of PSMILES and SMILES?
I would say that perhaps the beauty of dynamic tokenization is that one tokenizer for both would work comparably with using rigid tokenizers for each representation.
What are the reasons that make the authors think that dynamic tokenization would be beneficial to chemistry?
Right now, authors claim that the dynamic will be better. But is it compared to when one has rigid tokenizers for SMILES and another different for PSMILES?
Limitations:
There are some discussions about the limitations as future work challenges, which also include some things related to my concerns

Overall Recommendation: 1: Strong Reject: For instance, a paper with well-known results, unaddressed ethical considerations, or a poorly written paper where it is impossible to tell what the nature of its contribution is.
Confidence: 5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.

---

Official Review of Submission15667 by Reviewer wgYn
Official Reviewby Reviewer wgYn12 Mar 2026, 09:45 (modified: 24 Mar 2026, 14:30)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer wgYnRevisions
Summary:
This paper looks at whether a model can learn a better way to split chemical SMILES strings into meaningful pieces instead of relying on fixed tokenization rules. Using a hierarchical byte-level model called H-Net, the authors train several medium-sized models on both polymer and small-molecule datasets and show that the tokens learned are quite different depending on the chemical domain, while also becoming more efficient as training continues. They compare this learned tokenization with simpler character-level methods and SmilesPE, and find that the resulting representations can be useful for downstream property prediction, especially on some classification tasks, although the gains are less clear for regression. Overall, the paper is mainly a study of how end-to-end learned tokenization behaves for chemical strings, rather than a paper focused on achieving the best prediction performance.

Strengths And Weaknesses:
Strengths:
The paper studies a meaningful and not yet widely explored question: whether tokenization in chemical language models should be learned from data instead of being fixed beforehand. This is a worthwhile direction, and the idea that different chemical domains may develop their own useful vocabularies is easy to understand and potentially valuable for future work.
The paper also goes beyond raw performance by looking at what kinds of chemical fragments the model learns, such as whether token boundaries match atoms or functional groups, which makes the work more informative and scientifically useful than a paper that only reports benchmark results.
Weaknesses:
A main weakness of the paper is that its evidence for downstream usefulness is still quite limited and not strong enough to fully support the claims for a top venue like ICML. The evaluation uses frozen H-Net embeddings that are mean pooled and then fed into XGBoost, with comparison only against RDKit descriptors, which is a rather narrow set of baselines for a paper proposing a new representation-learning approach. More competitive comparisons with existing chemical language models or other modern learned molecular encoders are missing. Although the paper discusses models such as PolyBERT and ChemBERTa in the related work, these are not included in the main property-prediction experiments, which makes it harder to judge how strong the proposed representations really are.
The evidence for the paper’s significance is somewhat mixed. While the model does show real improvements over RDKit descriptors on the BBBP and HIV classification tasks, the gains are fairly modest, and its performance on regression tasks is consistently worse, in some cases by a large margin. In addition, BBBP and HIV are relatively well-studied and comparatively easier benchmark datasets, with many existing papers reporting substantially stronger results on these tasks. This makes it harder to view the reported downstream performance as strong evidence of broad practical impact.
The property prediction setup is not described in enough detail. Important aspects such as the data split strategy (for example, random, scaffold, or chronological), the source and size of the polymer datasets used for targets like Tg and MAC, and the steps taken to prevent data leakage are not clearly explained. This lack of detail makes it difficult to properly assess the reliability and fairness of the reported results.
Soundness: 2: fair
Presentation: 2: fair
Significance: 2: fair
Originality: 3: good
Key Questions For Authors:
Please provide a clear list of the state-of-the-art methods most relevant to your work, and explain why these methods were not included in your experimental comparisons.

Limitations:
The baseline methods are fairly weak, and the evaluation relies on relatively easy benchmark datasets, also the downstream results are not yet fully convincing.
Many stronger and more advanced relevant methods are missing from the comparisons.
Some important details of the experimental setup are not clearly described.
Overall Recommendation: 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility, incompletely addressed ethical considerations, or writing so poor that it is not possible to understand its key claims.
Confidence: 5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.

---

Official Review of Submission15667 by Reviewer Kh1S
Official Reviewby Reviewer Kh1S17 Feb 2026, 04:36 (modified: 24 Mar 2026, 14:30)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer Kh1SRevisions
Summary:
The paper introduces dynamic tokenization for SMILES using Hierarchical Networks (H-Net), showing that learned, data-driven segmentation adapts to chemical domain differences better than static tokenizers. Experiments on polymer and molecular datasets demonstrate that H-Net learns domain-specific vocabularies, captures chemically meaningful patterns, and produces embeddings that improve downstream classification performance, supporting dynamic tokenization as a promising alternative to s static chemical tokenization schemes.

Strengths And Weaknesses:
Strengths:

This paper focuses on SMILES tokenization, which is an important problem in cheminformatics.
The introduction of H-Net for dynamic tokenization of SMILES is interesting and interpretable.
Weaknesses:

The background claim that existing tokenization methods are all static is limited. Previous work [1] has explored using SMILES pair encoding to update the vocabulary regularly during training, which is also dynamic.
The key findings in the abstract (1) dataset specificity and (2) improved tokenization efficiency are somewhat naïve, as they are also expected under static SMILES pair encoding and are not necessarily unique advantages of dynamic tokenization.
The reported advantage of H-Net representations over RDKit descriptors is also limited. At a minimum, the paper should show that dynamic tokenization outperforms static atom-level tokenization and SMILES pair encoding.
The effect of SMILES randomization/enumeration is not discussed, which is very important in the context of SMILES tokenization.
[1] De novo Drug Design using Reinforcement Learning with Dynamic Vocabulary. Openreview.

Soundness: 2: fair
Presentation: 3: good
Significance: 2: fair
Originality: 3: good
Key Questions For Authors:
The tokenization method is expected to show its main effect in generative tasks rather than property prediction. Could this method be applied to molecular generation?
Why are some related works on SMILES tokenization [1,2] not discussed?
[1] Exploring data-driven chemical SMILES tokenization approaches to identify key protein–ligand binding moieties. Molecular Informatics, 2024.

[2] fragSMILES as a chemical string notation for advanced fragment and chirality representation. Communications Chemistry, 2025.

Limitations:
yes

Overall Recommendation: 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility, incompletely addressed ethical considerations, or writing so poor that it is not possible to understand its key claims.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
