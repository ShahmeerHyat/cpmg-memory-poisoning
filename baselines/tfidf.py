"""TF-IDF + logistic regression baselines (confound check), and the provenance-only leak check.

  tfidf_memory       : memory text only (pointwise)
  tfidf_full         : the same segments CPMG sees, flattened to text (provenance tags, source, query, neighbours)
  tfidf_memory_noasb : memory text, trained without ASB (for the cross-source ASB test)
  prov_only          : one-hot (channel, authority, parent) only -> provenance leak check
If tfidf_* is near-perfect, the data is too easy for that slice (see DECISIONS / report).
Usage: python -m baselines.tfidf
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from baselines.common import PRED_SPLITS, latency_ms, load, write_preds, write_run
from cpmg.features import as_text

SEED = 42


def run_text(name, to_text, exclude=()):
    train = load("train", exclude)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)
    X = vec.fit_transform([to_text(r) for r in train])
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=SEED)
    clf.fit(X, [r["label"] for r in train])
    f = lambda rows: clf.predict_proba(vec.transform([to_text(r) for r in rows]))[:, 1]
    for sp in PRED_SPLITS:
        rows = load(sp)
        write_preds(name, sp, rows, f(rows))
    write_run(name, latency_ms_bs1=latency_ms(f, load("test")), n_params=int(X.shape[1]) + 1, device="cpu",
              n_train=len(train), exclude_sources=list(exclude))
    print(name, "done")


def run_prov():
    cols = ["channel", "authority", "parent"]
    train = load("train")
    enc = OneHotEncoder(handle_unknown="ignore")
    X = enc.fit_transform([[r[c] for c in cols] for r in train])
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    clf.fit(X, [r["label"] for r in train])
    f = lambda rows: clf.predict_proba(enc.transform([[r[c] for c in cols] for r in rows]))[:, 1]
    for sp in PRED_SPLITS:
        rows = load(sp)
        write_preds("prov_only", sp, rows, f(rows))
    write_run("prov_only", latency_ms_bs1=latency_ms(f, load("test")), n_params=int(X.shape[1]) + 1, device="cpu",
              features=cols)
    print("prov_only done")


def main():
    np.random.seed(SEED)
    run_text("tfidf_memory", lambda r: r["memory"])
    run_text("tfidf_full", lambda r: as_text(r, "full"))
    run_text("tfidf_memory_noasb", lambda r: r["memory"], exclude=("ASB",))
    run_prov()


if __name__ == "__main__":
    main()
