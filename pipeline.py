from __future__ import annotations
import json, logging, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, GEOparse
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

warnings.filterwarnings("ignore")
logging.getLogger("GEOparse").setLevel(logging.ERROR)

TRAIN_GSE = "GSE123698"
EXTERNAL_GSE = ["GSE123696", "GSE123697"]
YOUNG_MAX, OLD_MIN, SEED = 40, 60, 42
DATA, OUT, MODELS = Path("data"), Path("results"), Path("models")
OUT.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
MODEL_PATH = MODELS / "immunosenescence_logreg.pkl"
KNOWN_MARKERS = {"CD28", "CD27", "CCR7", "KLRG1", "B3GAT1", "IL6", "TNF", "IL1B",
    "CDKN2A", "CDKN1A", "TP53", "GLB1", "GZMB", "IL7R", "SELL", "CCR6", "KLRB1",
    "HAVCR2", "PDCD1", "LAG3"}


def make_model():
    return Pipeline([("select", SelectKBest(f_classif, k=25)),
                     ("clf", LogisticRegression(C=0.01, max_iter=5000))])


def normalize(feats):
    z = StandardScaler().fit_transform(feats.values)
    return pd.DataFrame(np.nan_to_num(z), index=feats.index, columns=feats.columns)


def load_series(acc):
    gse = GEOparse.get_GEO(geo=acc, destdir=str(DATA), silent=True)
    ph = gse.phenotype_data
    df = gse.pivot_samples("VALUE").dropna(how="all").T
    df["age"] = pd.to_numeric(
        ph["characteristics_ch1.0.age"].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce").reindex(df.index)
    df["individual"] = ph["title"].astype(str).str.extract(r"(ind\d+)")[0].reindex(df.index)
    return df.dropna(subset=["age"])


def to_xy(df, genes=None):
    d = df[(df["age"] < YOUNG_MAX) | (df["age"] > OLD_MIN)]
    y = (d["age"] > OLD_MIN).astype(int).values
    feats = d.drop(columns=["age", "individual"])
    if genes is not None:
        feats = feats.reindex(columns=genes)
    return normalize(feats), y, list(feats.columns)


def probe_to_symbol():
    gpl = next(iter(GEOparse.get_GEO(geo=TRAIN_GSE, destdir=str(DATA),
                                     silent=True).gpls.values())).table
    out = {}
    for pid, sym in zip(gpl["ID"].astype(str), gpl["Gene Symbol"]):
        s = str(sym).split("///")[0].strip()
        if s not in ("", "---", "nan"):
            out[pid] = s
    return out


def main():
    log = []
    def say(m=""): print(m); log.append(m)

    say("=" * 60); say("DataNova Model 3 — immunosenescence classifier"); say("=" * 60)

    train = load_series(TRAIN_GSE)
    genes_all = train.drop(columns=["age", "individual"]).dropna(axis=1).columns.tolist()
    X, y, genes = to_xy(train, genes_all)
    say(f"\nTRAIN {TRAIN_GSE}: {len(y)} samples "
        f"({(y==0).sum()} younger, {(y==1).sum()} older), {len(genes)} genes.")
    say(f"Majority-class baseline: {max(y.mean(), 1-y.mean()):.3f}")

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    yp = cross_val_predict(make_model(), X, y, cv=cv)
    pr = cross_val_predict(make_model(), X, y, cv=cv, method="predict_proba")[:, 1]
    acc_in, auc_in = accuracy_score(y, yp), roc_auc_score(y, pr)
    say("\n[Internal 5-fold CV]")
    say(f"   accuracy : {acc_in:.3f}"); say(f"   ROC-AUC  : {auc_in:.3f}")
    say(f"   confusion (rows=true[young,old]):\n{confusion_matrix(y, yp)}")

    train_ids = set(train["individual"].dropna())
    ext = pd.concat([load_series(a) for a in EXTERNAL_GSE])
    ext = ext[~ext["individual"].isin(train_ids)].drop_duplicates("individual")
    Xext, yext, _ = to_xy(ext, genes)
    model = make_model().fit(X, y)
    yp_e, pr_e = model.predict(Xext), model.predict_proba(Xext)[:, 1]
    acc_ex, auc_ex = accuracy_score(yext, yp_e), roc_auc_score(yext, pr_e)
    say(f"\n[External validation on {len(yext)} UNSEEN individuals "
        f"from {'+'.join(EXTERNAL_GSE)}]")
    say(f"   {(yext==0).sum()} younger, {(yext==1).sum()} older")
    say(f"   accuracy : {acc_ex:.3f}"); say(f"   ROC-AUC  : {auc_ex:.3f}")
    say(f"   confusion (rows=true[young,old]):\n{confusion_matrix(yext, yp_e)}")

    p2s = probe_to_symbol()
    F, _ = f_classif(X.values, y)
    rank = pd.DataFrame({"probe": genes, "F": F})
    rank["symbol"] = rank["probe"].map(lambda p: p2s.get(str(p)))
    rank = rank.dropna(subset=["symbol"]).sort_values("F", ascending=False).drop_duplicates("symbol")
    rank["known"] = rank["symbol"].isin(KNOWN_MARKERS)
    rank.to_csv(OUT / "top_age_genes.csv", index=False)
    say("\n[Top 10 age-discriminating genes]")
    for i, r in enumerate(rank.head(10).itertuples(), 1):
        say(f"   {i:>2}. {r.symbol:<9} F={r.F:5.1f}" + ("   *known marker*" if r.known else ""))
    say(f"   known markers among strong genes: {', '.join(rank[rank['known']].head(8)['symbol'])}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, (title, yt, ypr) in zip(ax, [("Internal CV", y, pr),
                                        ("External (unseen people)", yext, pr_e)]):
        cm = confusion_matrix(yt, (ypr > 0.5).astype(int))
        a.imshow(cm, cmap="Blues")
        a.set_xticks([0, 1], ["young", "old"]); a.set_yticks([0, 1], ["young", "old"])
        a.set_xlabel("predicted"); a.set_ylabel("true")
        a.set_title(f"{title}\nacc={accuracy_score(yt,(ypr>0.5).astype(int)):.2f} "
                    f"AUC={roc_auc_score(yt,ypr):.2f}")
        for i in range(2):
            for j in range(2):
                a.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                       color="white" if cm[i, j] > cm.max()/2 else "black")
    fig.tight_layout(); fig.savefig(OUT / "validation_summary.png", dpi=140)

    joblib.dump({"model": model, "genes": genes,
                 "top_symbols": list(rank.head(25)["symbol"]),
                 "young_max": YOUNG_MAX, "old_min": OLD_MIN, "platform": "GPL15207"},
                MODEL_PATH)
    Path("metrics.json").write_text(json.dumps({
        "internal_cv": {"auc": round(auc_in, 4), "accuracy": round(acc_in, 4), "n": int(len(y))},
        "external": {"auc": round(auc_ex, 4), "accuracy": round(acc_ex, 4), "n": int(len(yext))},
        "baseline_accuracy": round(max(y.mean(), 1 - y.mean()), 4),
        "n_genes": len(genes), "k_selected": 25, "seed": SEED, "python": "3.14.6",
        "train_data": f"GEO {TRAIN_GSE}",
        "external_data": f"{'+'.join(EXTERNAL_GSE)} (unseen individuals only)",
    }, indent=2), encoding="utf-8")
    (OUT / "pipeline_report.txt").write_text("\n".join(log), encoding="utf-8")
    say(f"\nSaved: {MODEL_PATH}, metrics.json, results/")


if __name__ == "__main__":
    main()
