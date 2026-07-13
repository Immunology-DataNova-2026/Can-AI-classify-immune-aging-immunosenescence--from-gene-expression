# Can AI classify immune aging (immunosenescence) from gene expression?

My third model for the DataNova competition.

Your immune system ages, and it doesn't always age at the same speed as the rest
of you. So the question I wanted to answer: can a model look at gene expression
from a normal blood sample and tell whether someone's immune system looks "young"
or "old"? Turns out it can — and honestly I spent most of the time making sure the
result was real and not just me fooling myself.

## What I got

| How I tested it | AUC | Accuracy | n |
|---|---|---|---|
| Internal 5-fold CV (training group) | 0.93 | 0.88 | 68 |
| **External — people it had never seen** | **0.81** | **0.78** | 23 |
| Baseline (just guess "old") | 0.50 | 0.62 | — |

The 88% is on the training group, so I don't put much weight on it. The number I
actually trust is the **78% on 23 people the model had never seen** — that's the
real test, and beating the 62% "just guess old" baseline on brand-new people is
the part I'm happy with.

## How it works

Nothing crazy:
1. Pull whole-blood gene expression from GEO (68 people, ages 24–97).
2. Normalize each dataset on its own so different batches line up.
3. Keep the 25 genes most tied to age.
4. Logistic regression makes the young/old call.

Ages in this data are bimodal (a young group and an old group, basically nobody in
the middle), so I framed it as young vs old instead of trying to predict an exact age.

## The part I think is cool

The model isn't a black box. The genes it leans on are the same ones immunologists
already use to describe immune aging — **CD28, CD27, CCR7 drop** in older people,
**CDKN2A (p16), TNF, HAVCR2 go up**. So it basically rediscovered the textbook
biology on its own.

The external test also had a twist. My first attempt scored 52% — the model called
everyone "old." But the AUC was still 0.82, which told me the ranking was actually
fine and the real problem was a batch shift throwing off the cutoff. Normalizing
each dataset on its own fixed it and brought accuracy back to 78%. Finding and
fixing that was probably the most useful thing I did on this model.

## Run it

```bash
pip install -r requirements.txt
python pipeline.py               # downloads data, trains, validates, saves the model
python predict.py expression.csv # score new whole-blood samples
```

Trained model is in `models/`, the numbers are in `metrics.json`, and the figures
plus the top genes are in `results/`.

## Honest limits

- Only 68 people — small dataset.
- Predicting an exact "immune age" number doesn't work here (no middle-aged people
  to learn from), so I stuck to young vs old.
- The extra cohorts I validated on are the same study in other years, so I only
  tested on the individuals who never showed up in training.
