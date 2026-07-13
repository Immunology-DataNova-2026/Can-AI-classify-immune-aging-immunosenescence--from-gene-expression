from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path("models/immunosenescence_logreg.pkl")


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python predict.py expression.csv")
    if not MODEL_PATH.exists():
        sys.exit(f"model not found at {MODEL_PATH} — run `python pipeline.py` first")

    bundle = joblib.load(MODEL_PATH)
    model, genes = bundle["model"], bundle["genes"]
    df = pd.read_csv(sys.argv[1], index_col=0)

    if len(df) < 5:
        print(f"[warning] only {len(df)} sample(s); per-cohort normalization is "
              "unreliable below ~5 samples.\n")
    feats = df.reindex(columns=genes)
    missing = feats.isna().all(axis=0).sum()
    if missing:
        print(f"[warning] {missing}/{len(genes)} model genes absent from input "
              "(filled as 0).\n")

    z = np.nan_to_num(StandardScaler().fit_transform(feats.values))
    prob = model.predict_proba(z)[:, 1]
    print(pd.DataFrame({"sample": df.index, "P(older)": np.round(prob, 3),
                        "prediction": np.where(prob >= 0.5, "OLDER", "younger")}
                       ).to_string(index=False))


if __name__ == "__main__":
    main()
