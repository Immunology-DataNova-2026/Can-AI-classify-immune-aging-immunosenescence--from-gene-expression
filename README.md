# model-3 · Immunosenescence from Gene Expression ✅

**Can a model tell a "young" immune system from an "old" one from a blood gene-expression profile?**
**Honest result: AUC 0.93 internal, and it holds at AUC 0.81 / 78% accuracy on people the model has never seen.**

| Evaluation | AUC | Accuracy | n |
|---|---|---|---|
| **Internal 5-fold CV** (training cohort) | **0.93** | 0.88 | 68 |
| **External** (unseen individuals) | **0.81** | 0.78 | 23 |
| *Majority-class baseline* | 0.50 | 0.62 | — |

## Why it works
Immune aging (**immunosenescence**) leaves a reproducible transcriptional signature
in whole blood: loss of naive T-cells and co-stimulation (**CD28, CD27, CCR7 down**)
and rising cell-senescence + inflammation (**CDKN2A/p16, TNF, HAVCR2 up**). The model
selects the 25 most age-associated genes and classifies with logistic regression —
and the genes it picks are exactly these known markers, so it is not a black box.
Ages are bimodal (young 20s-30s vs old 60s-90s), so this is framed as young-vs-old
classification. 68 whole-blood samples, evaluated **by individual** (no leakage).

The external test is the real story: naively testing on the sibling cohorts gave
52% (the model called everyone "old") because of a **batch effect** — a per-dataset
distribution shift miscalibrated the 0.5 threshold, even though ranking (AUC 0.82)
still worked. Per-cohort z-scoring fixes it, recovering 78%. The "sibling" series
GSE123696/697 are the *same people* in other years, so the external set is
restricted to the 23 individuals who never appear in training.

## Run & deploy
```bash
python pipeline.py               # full pipeline -> AUC 0.81 external, saves the model
python predict.py expression.csv # score whole-blood samples (GPL15207 probe schema)
```
Model: `models/immunosenescence_logreg.pkl`. Metrics: `metrics.json`. Figures & driver
genes in `results/`. Data auto-downloads from GEO (GSE123698 train; GSE123696/697 external).
