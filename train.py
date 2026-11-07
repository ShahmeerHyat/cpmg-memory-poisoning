"""Train a CPMG (or ablation / content-only) DeBERTa-v3-base risk scorer and write predictions.

Usage: python train.py --config configs/cpmg_full.yaml [--limit 200]
Writes: checkpoints/<name>/ (best by val AUROC), results/logs/<name>_train.jsonl (loss),
        results/runs/<name>.json (config + val metrics + timings),
        results/preds/<name>/<split>.jsonl (id, score) for val and every test set.
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from transformers import get_linear_schedule_with_warmup

from cpmg.features import encode
from cpmg.model import collate, load_model, load_tokenizer, score

ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL_SPLITS = ["val", "test", "test_asb_xsource", "test_notinject"]


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def load_split(name, exclude_sources=(), limit=None, seed=42):
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data/processed", f"{name}.jsonl"))]
    rows = [r for r in rows if r["source_dataset"] not in exclude_sources]
    if limit:
        rows = random.Random(seed).sample(rows, min(limit, len(rows)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: N train / N//2 eval examples")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    name = cfg["name"] + a.suffix
    set_seed(cfg["seed"])
    dev = torch.device("cuda")
    excl = cfg.get("exclude_sources", [])

    tok = load_tokenizer()
    train = load_split("train", excl, a.limit, cfg["seed"])
    val = load_split("val", excl, a.limit // 2 if a.limit else None, cfg["seed"])
    enc = lambda rows: [encode(r, tok, cfg["variant"], cfg["max_len"]) for r in rows]
    tr_ids, va_ids = enc(train), enc(val)
    tr_lab = [r["label"] for r in train]
    print(f"[{name}] train {len(train)} (pos {sum(tr_lab)}), val {len(val)}; "
          f"mean len {np.mean([len(x) for x in tr_ids]):.0f}, max {max(len(x) for x in tr_ids)}", flush=True)

    model = load_model(tok, grad_ckpt=cfg.get("grad_ckpt", False)).to(dev)
    n_pos = sum(tr_lab)
    w = torch.tensor([len(tr_lab) / (2 * (len(tr_lab) - n_pos)), len(tr_lab) / (2 * n_pos)], dtype=torch.float, device=dev)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    bs, acc = cfg["batch_size"], cfg["grad_accum"]
    steps_per_epoch = math.ceil(len(train) / bs / acc)
    total = steps_per_epoch * cfg["epochs"]
    sch = get_linear_schedule_with_warmup(opt, int(cfg["warmup"] * total), total)

    os.makedirs(os.path.join(ROOT, "results/logs"), exist_ok=True)
    log = open(os.path.join(ROOT, "results/logs", f"{name}_train.jsonl"), "w")
    ckpt = os.path.join(ROOT, "checkpoints", name)
    best, step, t0 = -1.0, 0, time.time()
    history = []
    for ep in range(cfg["epochs"]):
        model.train()
        order = list(range(len(train)))
        random.Random(cfg["seed"] + ep).shuffle(order)
        run_loss, n_b = 0.0, 0
        for bi, s in enumerate(range(0, len(order), bs)):
            idx = order[s: s + bs]
            b = collate([{"ids": tr_ids[i], "label": tr_lab[i]} for i in idx], tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=b["input_ids"].to(dev), attention_mask=b["attention_mask"].to(dev)).logits
            loss = loss_fn(logits.float(), b["labels"].to(dev)) / acc
            loss.backward()
            run_loss += loss.item() * acc
            n_b += 1
            if (bi + 1) % acc == 0 or s + bs >= len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sch.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    log.write(json.dumps({"step": step, "epoch": ep + step / max(1, steps_per_epoch) - ep,
                                          "loss": run_loss / n_b, "lr": sch.get_last_lr()[0]}) + "\n")
                    log.flush()
                    run_loss, n_b = 0.0, 0
                if step % 100 == 0:
                    print(f"[{name}] ep {ep} step {step}/{total} {time.time()-t0:.0f}s", flush=True)
        vs = score(model, va_ids, tok.pad_token_id, dev)
        auc = roc_auc_score([r["label"] for r in val], vs)
        history.append({"epoch": ep, "val_auroc": auc, "elapsed_s": time.time() - t0})
        print(f"[{name}] epoch {ep} val AUROC {auc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if auc > best:
            best = auc
            model.save_pretrained(ckpt)
            tok.save_pretrained(ckpt)
    train_s = time.time() - t0
    log.close()

    # predictions with the best checkpoint
    del model
    torch.cuda.empty_cache()
    model = load_model(tok, path=ckpt).to(dev)
    pred_dir = os.path.join(ROOT, "results/preds", name)
    os.makedirs(pred_dir, exist_ok=True)
    for sp in EVAL_SPLITS:
        rows = val if sp == "val" else load_split(sp, (), a.limit // 2 if a.limit else None, cfg["seed"])
        sc = score(model, enc(rows), tok.pad_token_id, dev)
        with open(os.path.join(pred_dir, f"{sp}.jsonl"), "w") as f:
            for r, v in zip(rows, sc):
                f.write(json.dumps({"id": r["id"], "score": v}) + "\n")
    # latency per check (batch size 1) on 200 test examples
    rows = load_split("test", (), None, cfg["seed"])[:200]
    ids = enc(rows)
    score(model, ids[:10], tok.pad_token_id, dev, batch_size=1)
    torch.cuda.synchronize()
    t1 = time.time()
    score(model, ids, tok.pad_token_id, dev, batch_size=1)
    torch.cuda.synchronize()
    lat_ms = (time.time() - t1) / len(ids) * 1000
    n_params = sum(p.numel() for p in model.parameters())
    os.makedirs(os.path.join(ROOT, "results/runs"), exist_ok=True)
    json.dump(dict(name=name, config=cfg, limit=a.limit, n_train=len(train), n_val=len(val), best_val_auroc=best,
                   history=history, train_seconds=train_s, latency_ms_bs1=lat_ms, n_params=n_params,
                   gpu=torch.cuda.get_device_name(0), torch=torch.__version__),
              open(os.path.join(ROOT, "results/runs", f"{name}.json"), "w"), indent=1)
    print(f"[{name}] done. best val AUROC {best:.4f}, train {train_s:.0f}s, latency {lat_ms:.1f} ms", flush=True)


if __name__ == "__main__":
    main()
