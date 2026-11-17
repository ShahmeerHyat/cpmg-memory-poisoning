import json
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = ["train", "val", "test", "test_asb_xsource", "test_notinject"]
PRED_SPLITS = ["val", "test", "test_asb_xsource", "test_notinject"]
# extra test files with the same ids as test.jsonl (paraphrase attack, D31), scored when present
PRED_SPLITS += [x for x in ("test@para", "test@paranq")
                if os.path.exists(os.path.join(ROOT, "data/processed", f"{x}.jsonl"))]


def load(split, exclude_sources=()):
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data/processed", f"{split}.jsonl"))]
    return [r for r in rows if r["source_dataset"] not in exclude_sources]


def write_preds(name, split, rows, scores, scores_act=None):
    """score = poison score; score_act = activation score (defaults to the same score for single-score baselines)."""
    d = os.path.join(ROOT, "results/preds", name)
    os.makedirs(d, exist_ok=True)
    scores_act = scores if scores_act is None else scores_act
    with open(os.path.join(d, f"{split}.jsonl"), "w") as f:
        for r, s, a in zip(rows, scores, scores_act):
            f.write(json.dumps({"id": r["id"], "score": float(s), "score_act": float(a)}) + "\n")


def latency_ms(fn, rows, n=200):
    rows = rows[:n]
    fn(rows[:5])
    t = time.time()
    for r in rows:
        fn([r])
    return (time.time() - t) / len(rows) * 1000


def write_run(name, **kw):
    d = os.path.join(ROOT, "results/runs")
    os.makedirs(d, exist_ok=True)
    json.dump(dict(name=name, **kw), open(os.path.join(d, f"{name}.json"), "w"), indent=1)
