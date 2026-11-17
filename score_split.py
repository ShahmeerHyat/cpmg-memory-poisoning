"""Score extra evaluation files with already-trained checkpoints (no training).

Usage: python score_split.py --splits test@para test@paranq [--models cpmg_full cpmg_full_s43 ...]
Reads data/processed/<split>.jsonl, writes results/preds/<model>/<split>.jsonl (id, score, score_act).
The input variant of each model comes from its results/runs/<model>.json config.
"""
import argparse
import json
import os

import torch

from cpmg.features import encode
from cpmg.model import load_model, load_tokenizer, score

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--models", nargs="*", default=None)
    a = ap.parse_args()
    runs = os.path.join(ROOT, "results/runs")
    models = a.models or sorted(f[:-5] for f in os.listdir(runs)
                                if os.path.isdir(os.path.join(ROOT, "checkpoints", f[:-5])))
    dev = torch.device("cuda")
    for m in models:
        cfg = json.load(open(os.path.join(runs, f"{m}.json")))["config"]
        ckpt = os.path.join(ROOT, "checkpoints", m)
        tok = load_tokenizer(ckpt)
        model = load_model(tok, path=ckpt).to(dev)
        for sp in a.splits:
            rows = [json.loads(l) for l in open(os.path.join(ROOT, "data/processed", f"{sp}.jsonl"))]
            p, act = score(model, [encode(r, tok, cfg["variant"], cfg["max_len"]) for r in rows], tok.pad_token_id, dev)
            os.makedirs(os.path.join(ROOT, "results/preds", m), exist_ok=True)
            with open(os.path.join(ROOT, "results/preds", m, f"{sp}.jsonl"), "w") as f:
                for r, x, y in zip(rows, p, act):
                    f.write(json.dumps({"id": r["id"], "score": x, "score_act": y}) + "\n")
            print(m, sp, len(rows), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
