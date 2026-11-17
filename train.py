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

from cpmg.features import VARIANTS, encode, swap_prov
from cpmg.model import collate, load_model, load_tokenizer, score

ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL_SPLITS = ["val", "test", "test_asb_xsource", "test_notinject"]


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def load_split(name, exclude_sources=(), limit=None, seed=42):
    """exclude_sources entries match a record's source_dataset or its subset."""
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data/processed", f"{name}.jsonl"))]
    rows = [r for r in rows if r["source_dataset"] not in exclude_sources and r["subset"] not in exclude_sources]
    if limit:
        rows = random.Random(seed).sample(rows, min(limit, len(rows)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: N train / N//2 eval examples")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--seed", type=int, default=None, help="override the config seed; run is named <name>_s<seed>")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.seed is not None and a.seed != cfg["seed"]:
        cfg["seed"] = a.seed
        a.suffix += f"_s{a.seed}"
    name = cfg["name"] + a.suffix
    set_seed(cfg["seed"])
    dev = torch.device("cuda")
    excl = cfg.get("exclude_sources", [])

    tok = load_tokenizer()
    train = load_split("train", excl, a.limit, cfg["seed"])
    val = load_split("val", excl, a.limit // 2 if a.limit else None, cfg["seed"])
    enc = lambda rows: [encode(r, tok, cfg["variant"], cfg["max_len"]) for r in rows]
    tr_ids, va_ids = enc(train), enc(val)
    # two targets per example: [poison, active]; active = -1 where it is not defined (masked in the loss)
    tr_y = [[r["y_poison"], r["y_active"]] for r in train]
    p_drop = cfg.get("prov_dropout", 0.0) if VARIANTS[cfg["variant"]]["prov"] else 0.0
    tr_ids_np = [encode(r, tok, cfg["variant"], cfg["max_len"], drop_prov=True) for r in train] if p_drop else None
    tr_lab = [y[0] for y in tr_y]
    print(f"[{name}] train {len(train)} (poison {sum(tr_lab)}, active {sum(1 for y in tr_y if y[1] == 1)} of "
          f"{sum(1 for y in tr_y if y[1] >= 0)} defined), val {len(val)}; "
          f"mean len {np.mean([len(x) for x in tr_ids]):.0f}, max {max(len(x) for x in tr_ids)}", flush=True)

    model = load_model(tok, grad_ckpt=cfg.get("grad_ckpt", False)).to(dev)
    Y = np.array(tr_y, dtype=float)
    # weighted BCE per output: pos_weight = (#neg / #pos) among the examples where that target is defined
    pw = torch.tensor([((Y[:, k] == 0).sum() / max(1, (Y[:, k] == 1).sum())) for k in (0, 1)], dtype=torch.float, device=dev)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")

    def loss_fn(logits, y):
        m = (y >= 0).float()
        return (bce(logits, y.clamp(min=0)) * m).sum() / m.sum().clamp(min=1)

    def val_score(rows, sc_p, sc_a):
        yp, ya = np.array([r["y_poison"] for r in rows]), np.array([r["y_active"] for r in rows])
        out = {"poison": roc_auc_score(yp, sc_p)}
        d = ya >= 0
        if d.any() and 0 < ya[d].sum() < d.sum():
            out["active"] = roc_auc_score(ya[d], np.array(sc_a)[d])
        out["mean"] = float(np.mean(list(out.values())))
        return out
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    bs, acc = cfg["batch_size"], cfg["grad_accum"]
    steps_per_epoch = math.ceil(len(train) / bs / acc)
    total = steps_per_epoch * cfg["epochs"]
    sch = get_linear_schedule_with_warmup(opt, int(cfg["warmup"] * total), total)

    os.makedirs(os.path.join(ROOT, "results/logs"), exist_ok=True)
    ckpt = os.path.join(ROOT, "checkpoints", name)
    os.makedirs(ckpt, exist_ok=True)
    resume_path = os.path.join(ckpt, "resume.pt")
    stop_flag = os.path.join(ROOT, "results", "STOP")
    best, step, t0 = -1.0, 0, time.time()
    history, start_ep, start_bi = [], 0, 0
    if os.path.exists(resume_path) and not a.limit:
        st = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sch.load_state_dict(st["sch"])
        best, step, history, start_ep, start_bi = st["best"], st["step"], st["history"], st["ep"], st["bi"]
        t0 = time.time() - st["elapsed"]
        del st
        print(f"[{name}] RESUMED at epoch {start_ep}, batch {start_bi}, step {step}/{total}", flush=True)
    log = open(os.path.join(ROOT, "results/logs", f"{name}_train.jsonl"), "a" if step else "w")

    def save_resume(ep_, bi_):
        tmp = resume_path + ".tmp"
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(), best=best, step=step,
                        history=history, ep=ep_, bi=bi_, elapsed=time.time() - t0), tmp)
        os.replace(tmp, resume_path)

    for ep in range(start_ep, cfg["epochs"]):
        model.train()
        order = list(range(len(train)))
        ep_rng = random.Random(cfg["seed"] + ep)
        ep_rng.shuffle(order)
        ep_ids = [tr_ids_np[i] if (p_drop and ep_rng.random() < p_drop) else tr_ids[i] for i in range(len(train))]
        run_loss, n_b = 0.0, 0
        for bi, s in enumerate(range(0, len(order), bs)):
            if ep == start_ep and bi < start_bi:
                continue
            idx = order[s: s + bs]
            b = collate([{"ids": ep_ids[i], "y": tr_y[i]} for i in idx], tok.pad_token_id)
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
                stop = os.path.exists(stop_flag)
                if (step % cfg.get("save_every", 150) == 0 or stop) and not a.limit:
                    save_resume(ep, bi + 1)
                    if stop:
                        print(f"[{name}] STOP requested: state saved at step {step}/{total}. Rerun to resume.", flush=True)
                        raise SystemExit(3)
        vp, va_ = score(model, va_ids, tok.pad_token_id, dev)
        vm = val_score(val, vp, va_)
        auc = vm["mean"]  # model selection: mean of the val AUROCs of the two outputs
        history.append({"epoch": ep, "val_auroc": auc, "val": vm, "elapsed_s": time.time() - t0})
        print(f"[{name}] epoch {ep} val AUROC {vm} ({time.time()-t0:.0f}s)", flush=True)
        if auc > best:
            best = auc
            model.save_pretrained(ckpt)
            tok.save_pretrained(ckpt)
        if not a.limit:
            save_resume(ep + 1, 0)
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
        jobs = [(sp, enc(rows))]
        if VARIANTS[cfg["variant"]]["prov"]:  # D25: provenance removed, and provenance swapped (content unchanged)
            jobs.append((sp + "@noprov", [encode(r, tok, cfg["variant"], cfg["max_len"], drop_prov=True) for r in rows]))
            if sp == "test":
                jobs.append((sp + "@provswap", [encode(swap_prov(r), tok, cfg["variant"], cfg["max_len"]) for r in rows]))
        for jn, ids_ in jobs:
            sc_p, sc_a = score(model, ids_, tok.pad_token_id, dev)
            with open(os.path.join(pred_dir, f"{jn}.jsonl"), "w") as f:
                for r, v, w_ in zip(rows, sc_p, sc_a):
                    f.write(json.dumps({"id": r["id"], "score": v, "score_act": w_}) + "\n")
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
                   gpu=torch.cuda.get_device_name(0), torch=torch.__version__, data_version="v2"),
              open(os.path.join(ROOT, "results/runs", f"{name}.json"), "w"), indent=1)
    if os.path.exists(resume_path):
        os.remove(resume_path)
    print(f"[{name}] done. best val AUROC {best:.4f}, train {train_s:.0f}s, latency {lat_ms:.1f} ms", flush=True)


if __name__ == "__main__":
    main()
