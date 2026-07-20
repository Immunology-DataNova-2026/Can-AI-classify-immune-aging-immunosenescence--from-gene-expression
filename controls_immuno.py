from __future__ import annotations
import json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

import pipeline as P

warnings.filterwarnings("ignore")
OUT = Path("results"); OUT.mkdir(exist_ok=True)
SEED = 42
rng = np.random.default_rng(SEED)
log: list[str] = []


def say(m=""):
    print(m)
    log.append(m)


say("# Control experiments\n")
say("Loading GEO series (cached after the first run)...\n")
train = P.load_series(P.TRAIN_GSE)
genes_all = train.drop(columns=["age", "individual"]).dropna(axis=1).columns.tolist()
X, y, genes = P.to_xy(train, genes_all)

train_ids = set(train["individual"].dropna())
ext = pd.concat([P.load_series(a) for a in P.EXTERNAL_GSE])
ext = ext[~ext["individual"].isin(train_ids)].drop_duplicates("individual")
Xext, yext, _ = P.to_xy(ext, genes)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

say(f"- training samples: {len(y)}  ({(y==0).sum()} younger, {(y==1).sum()} older)")
say(f"- external samples: {len(yext)}  ({(yext==0).sum()} younger, {(yext==1).sum()} older)")
say(f"- candidate genes: {len(genes):,}\n")


def fit_score(Xtr, ytr, Xte, yte, cols=None):
    if cols is None:
        model = P.make_model()
    else:
        idx = [genes.index(c) for c in cols]
        Xtr, Xte = Xtr.iloc[:, idx], Xte.iloc[:, idx]
        model = Pipeline([("clf", LogisticRegression(C=0.01, max_iter=5000))])
    model.fit(Xtr, ytr)
    pr = model.predict_proba(Xte)[:, 1]
    return accuracy_score(yte, (pr > .5).astype(int)), roc_auc_score(yte, pr)


say("## CONTROL 1 — random-gene null distribution\n")
say("Does the ANOVA filter actually do work, or would any 25 genes score this well?\n")
N_DRAWS = 100
rows = []
for i in range(N_DRAWS):
    cols = list(rng.choice(genes, size=25, replace=False))
    acc, auc = fit_score(X, y, Xext, yext, cols)
    rows.append({"draw": i, "accuracy": acc, "auc": auc})
rand = pd.DataFrame(rows)
rand.to_csv(OUT / "control_random_genes.csv", index=False)

real_acc, real_auc = 0.7826, 0.8106
pct = (rand["auc"] < real_auc).mean() * 100
say(f"- random 25-gene panels, external AUC: mean {rand['auc'].mean():.3f}, "
    f"sd {rand['auc'].std():.3f}, range {rand['auc'].min():.3f}-{rand['auc'].max():.3f}")
say(f"- random 25-gene panels, external accuracy: mean {rand['accuracy'].mean():.3f}")
say(f"- **selected panel: AUC {real_auc:.3f} — above {pct:.0f}% of random draws**")
say(f"- empirical p-value: {(rand['auc'] >= real_auc).mean():.3f}\n")


say("## CONTROL 2 — how much of the signal lives in the top gene?\n")
F_scores, _ = f_classif(X.values, y)
order = np.argsort(F_scores)[::-1]
ranked = [genes[i] for i in order]
rows = []
for k in (1, 3, 5, 10, 25, 50, 100):
    acc, auc = fit_score(X, y, Xext, yext, ranked[:k])
    rows.append({"k": k, "genes": ", ".join(ranked[:min(k, 3)]) + ("..." if k > 3 else ""),
                 "ext_accuracy": round(acc, 4), "ext_auc": round(auc, 4)})
    say(f"- top {k:>3} genes: external acc {acc:.3f}, AUC {auc:.3f}")
pd.DataFrame(rows).to_csv(OUT / "control_panel_size.csv", index=False)
say("\n(Selection here is fit on the full training set, matching how the final "
    "model is fitted before external validation.)\n")


say("## CONTROL 3 — what is refitting the filter inside each fold worth?\n")

correct = cross_val_predict(P.make_model(), X, y, cv=cv, method="predict_proba")[:, 1]
acc_ok, auc_ok = accuracy_score(y, (correct > .5).astype(int)), roc_auc_score(y, correct)

leaky_sel = SelectKBest(f_classif, k=25).fit(X, y)
X_leak = pd.DataFrame(leaky_sel.transform(X))
leaked = cross_val_predict(LogisticRegression(C=0.01, max_iter=5000),
                           X_leak, y, cv=cv, method="predict_proba")[:, 1]
acc_bad, auc_bad = accuracy_score(y, (leaked > .5).astype(int)), roc_auc_score(y, leaked)

say(f"- filter refit inside each fold (correct): acc {acc_ok:.3f}, AUC {auc_ok:.3f}")
say(f"- filter fit once on all data (leaky):     acc {acc_bad:.3f}, AUC {auc_bad:.3f}")
say(f"- **inflation from selection leakage: {auc_bad - auc_ok:+.3f} AUC**\n")
say("Paste into Table 7, row 'Filter refit inside folds'.\n")


say("## CONTROL 4 — empirical ROC and PR curves\n")
model = P.make_model().fit(X, y)
pr_ext = model.predict_proba(Xext)[:, 1]
curves = {}
for name, yt, pp in (("internal_cv", y, correct), ("external", yext, pr_ext)):
    fpr, tpr, _ = roc_curve(yt, pp)
    prec, rec, _ = precision_recall_curve(yt, pp)
    curves[name] = {"auc": float(roc_auc_score(yt, pp)),
                    "ap": float(average_precision_score(yt, pp)),
                    "roc": [[float(a), float(b)] for a, b in zip(fpr, tpr)],
                    "pr": [[float(a), float(b)] for a, b in zip(rec, prec)]}
    say(f"- {name}: AUC {curves[name]['auc']:.3f}, average precision {curves[name]['ap']:.3f}, "
        f"{len(fpr)} ROC points")
(OUT / "curves.json").write_text(json.dumps(curves, indent=2), encoding="utf-8")
say("\nWritten to results/curves.json — plot these instead of the schematic in Figure 4.\n")


say("## CONTROL 5 — bootstrap confidence intervals\n")
for name, yt, pp in (("internal_cv", y, correct), ("external", yext, pr_ext)):
    aucs = []
    for _ in range(2000):
        idx = rng.integers(0, len(yt), len(yt))
        if len(np.unique(yt[idx])) == 2:
            aucs.append(roc_auc_score(yt[idx], pp[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    say(f"- {name}: AUC {roc_auc_score(yt, pp):.3f}, 95% CI [{lo:.3f}, {hi:.3f}] "
        f"({len(aucs)} valid resamples)")

(OUT / "controls_report.md").write_text("\n".join(log), encoding="utf-8")
say(f"\nSaved: {OUT/'controls_report.md'} + 3 CSV/JSON artifacts")
