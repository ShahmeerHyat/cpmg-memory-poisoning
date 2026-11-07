import json
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = ["train", "val", "test", "test_asb_xsource", "test_notinject"]
PRED_SPLITS = ["val", "test", "test_asb_xsource", "test_notinject"]


def load(split, exclude_sources=()):
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data/processed", f"{split}.jsonl"))]
    return [r for r in rows if r["source_dataset"] not in exclude_sources]


def write_preds(name, split, rows, scores):
    d = os.path.join(ROOT, "results/preds", name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{split}.jsonl"), "w") as f:
        for r, s in zip(rows, scores):
            f.write(json.dumps({"id": r["id"], "score": float(s)}) + "\n")


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
