"""Evaluate every model from its prediction files and write metrics, tables and plots.

Inputs:  data/processed/*.jsonl, results/preds/<model>/<split>.jsonl, results/runs/*.json, results/logs/*_train.jsonl
Outputs: results/metrics.csv, results/metrics.json, results/tables.md, results/policy.json,
         results/plots/{roc_test,roc_by_level,training_loss,auroc_by_level}.png
Thresholds are chosen on val only (max F1 on val), then applied unchanged to every test set.
Usage: python eval.py
"""
import collections
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score, roc_curve

from cpmg import policy

ROOT = os.path.dirname(os.path.abspath(__file__))
R = lambda *a: os.path.join(ROOT, "results", *a)
SEED = 42

MODELS = collections.OrderedDict([
    ("cpmg_full", "CPMG (retrieval checkpoint)"),
    ("cpmg_write_retrieval", "CPMG write + retrieval"),
    ("write_only", "CPMG write-time only"),
    ("abl_no_prov", "CPMG w/o provenance"),
    ("abl_no_query", "CPMG w/o query/action"),
    ("abl_no_nbrs", "CPMG w/o neighbours"),
    ("content_only", "Content-only DeBERTa"),
    ("piguard", "PIGuard (zero-shot)"),
    ("keyword", "Keyword filter"),
    ("tfidf_memory", "TF-IDF+LR (memory)"),
    ("tfidf_full", "TF-IDF+LR (full input)"),
    ("prov_only", "Provenance-only LR (leak check)"),
    ("cpmg_noasb", "CPMG trained w/o ASB"),
    ("tfidf_memory_noasb", "TF-IDF+LR (memory) w/o ASB"),
])
# Models whose training never saw ASB (valid on the cross-source ASB test)
XSOURCE_VALID = {"cpmg_noasb", "tfidf_memory_noasb", "piguard", "keyword"}

# Palette: reference categorical order (dataviz skill), light mode
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (8, 2))]


def jl(p):
    return [json.loads(l) for l in open(p)]


def load_data():
    return {sp: {r["id"]: r for r in jl(os.path.join(ROOT, "data/processed", f"{sp}.jsonl"))}
            for sp in ("val", "test", "test_asb_xsource", "test_notinject")}


def load_preds(models, data):
    preds = {}
    for m in models:
        if m == "cpmg_write_retrieval":
            continue
        d = R("preds", m)
        if not os.path.isdir(d):
            continue
        preds[m] = {sp: {r["id"]: r["score"] for r in jl(os.path.join(d, f"{sp}.jsonl"))}
                    for sp in data if os.path.exists(os.path.join(d, f"{sp}.jsonl"))}
    if "cpmg_full" in preds and "write_only" in preds:  # flag if either checkpoint flags
        preds["cpmg_write_retrieval"] = {
            sp: {i: max(preds["cpmg_full"][sp][i], preds["write_only"][sp][i])
                 for i in preds["cpmg_full"][sp] if i in preds["write_only"].get(sp, {})}
            for sp in preds["cpmg_full"]}
    return preds


def best_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p * r / np.maximum(p + r, 1e-12)
    i = int(np.nanargmax(f[:-1])) if len(t) else 0
    return float(t[i]) if len(t) else 0.5


def tpr_at_fpr(y, s, target):
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= target + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def boot_auc(y, s, n=1000):
    rng = np.random.default_rng(SEED)
    y, s = np.asarray(y), np.asarray(s)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        out.append(roc_auc_score(y[i], s[i]))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (None, None)


def metrics(rows, scores, thr, ci=False):
    y = np.array([r["label"] for r in rows])
    s = np.array(scores)
    yhat = (s >= thr).astype(int)
    twins = np.array([r["variant"].endswith("_twin") for r in rows])
    neg = y == 0
    m = dict(n=int(len(y)), n_pos=int(y.sum()), n_neg=int(neg.sum()), threshold=thr,
             fpr_all_neg=float(yhat[neg].mean()) if neg.any() else None,
             frr_twins=float(yhat[twins & neg].mean()) if (twins & neg).any() else None,
             n_twins=int((twins & neg).sum()))
    if 0 < y.sum() < len(y):
        m.update(auroc=float(roc_auc_score(y, s)), tpr_at_1fpr=tpr_at_fpr(y, s, 0.01),
                 tpr_at_5fpr=tpr_at_fpr(y, s, 0.05), f1=float(f1_score(y, yhat, zero_division=0)),
                 recall=float(yhat[y == 1].mean()),
                 precision=float((y[yhat == 1] == 1).mean()) if yhat.sum() else 0.0)
        if ci and y.sum() >= 5 and neg.sum() >= 5:
            m["auroc_ci95"] = boot_auc(y, s)
    elif y.sum() == len(y) and len(y):
        m.update(recall=float(yhat.mean()))
    return m


def eval_sets(data):
    t = list(data["test"].values())
    sets = collections.OrderedDict()
    sets["test_all"] = ("test", t)
    for lvl in ("L1", "L2", "L3"):
        sets[lvl] = ("test", [r for r in t if r["level"] == lvl])
    sets["heldout_family"] = ("test", [r for r in t if r["group_family"] == "badchain"])
    for lvl in ("L1", "L2", "L3"):
        sets[f"heldout_{lvl}"] = ("test", [r for r in t if r["group_family"] == "badchain" and r["level"] == lvl])
    sets["seen_family_ap"] = ("test", [r for r in t if r["group_family"] == "agentpoison_ap"])
    for lvl in ("L1", "L2", "L3"):
        sets[f"seen_{lvl}"] = ("test", [r for r in t if r["group_family"] == "agentpoison_ap" and r["level"] == lvl])
    for src in ("AgentPoison", "ASB", "memshield-bench", "MemoryGraft"):
        sets[f"src_{src}"] = ("test", [r for r in t if r["source_dataset"] == src])
    sets["asb_xsource"] = ("test_asb_xsource", list(data["test_asb_xsource"].values()))
    sets["notinject"] = ("test_notinject", list(data["test_notinject"].values()))
    return sets


def fmt(v, pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{100*v:.1f}" if pct else f"{v:.3f}"


def table(title, res, models, sets, cols, note=""):
    out = [f"\n### {title}\n"]
    if note:
        out.append(f"{note}\n")
    hdr = ["Model"] + [f"{s} {c}" for s in sets for c, _ in cols]
    out.append("| " + " | ".join(hdr) + " |")
    out.append("|" + "---|" * len(hdr))
    for m in models:
        if m not in res:
            continue
        cells = [MODELS[m]]
        for s in sets:
            mm = res[m].get(s, {})
            for c, pct in cols:
                cells.append(fmt(mm.get(c), pct))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_roc(ax, sets_rows, preds, models, split, title):
    style_ax(ax)
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1, linestyle="--")
    for k, m in enumerate(models):
        if m not in preds or split not in preds[m]:
            continue
        y = [r["label"] for r in sets_rows]
        s = [preds[m][split][r["id"]] for r in sets_rows]
        if len(set(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=SERIES[k], linestyle=STYLES[k], linewidth=2,
                label=f"{MODELS[m]} ({roc_auc_score(y, s):.3f})")
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel("False positive rate", fontsize=8, color=INK2)
    ax.set_ylabel("True positive rate", fontsize=8, color=INK2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right", labelcolor=INK)


def policy_eval(data):
    """Authority policy on the real (non-derived) L1 records of the test and ASB cross-source sets."""
    out = {}
    for name, sp in (("test", "test"), ("asb_xsource", "test_asb_xsource")):
        rows = [r for r in data[sp].values() if r["level"] in ("L1", "bg")]
        agg = collections.defaultdict(lambda: collections.Counter())
        for r in rows:
            dec, cat = policy.decide(r["authority"], r["memory"])
            key = f"{r['source_dataset']}|label={r['label']}|auth={r['authority']}"
            agg[key]["records"] += 1
            agg[key]["protected_action"] += cat is not None
            agg[key]["blocked"] += dec == "block"
        tot = collections.Counter()
        for k, c in agg.items():
            lab = k.split("|")[1]
            for kk, v in c.items():
                tot[f"{lab}|{kk}"] += v
        out[name] = dict(by_source={k: dict(v) for k, v in sorted(agg.items())}, totals=dict(tot))
    return out


def main():
    os.makedirs(R("plots"), exist_ok=True)
    data = load_data()
    preds = load_preds(MODELS, data)
    models = [m for m in MODELS if m in preds]
    sets = eval_sets(data)
    res, thr = {}, {}
    for m in models:
        v = [r for r in data["val"].values() if r["id"] in preds[m]["val"]]
        thr[m] = best_threshold([r["label"] for r in v], [preds[m]["val"][r["id"]] for r in v])
        res[m] = {"val": metrics(v, [preds[m]["val"][r["id"]] for r in v], thr[m])}
        for sname, (sp, rows) in sets.items():
            if sname == "asb_xsource" and m not in XSOURCE_VALID:
                continue
            if sp not in preds[m]:
                continue
            rows = [r for r in rows if r["id"] in preds[m][sp]]
            if not rows:
                continue
            ci = sname in ("test_all", "L1", "L2", "L3", "heldout_family", "asb_xsource")
            res[m][sname] = metrics(rows, [preds[m][sp][r["id"]] for r in rows], thr[m], ci=ci)

    runs = {os.path.basename(p)[:-5]: json.load(open(R("runs", p))) for p in os.listdir(R("runs")) if p.endswith(".json")}
    pol = policy_eval(data)
    json.dump(pol, open(R("policy.json"), "w"), indent=1)
    json.dump(dict(thresholds=thr, results=res, runs={k: {kk: v.get(kk) for kk in
               ("latency_ms_bs1", "n_params", "train_seconds", "best_val_auroc", "history")} for k, v in runs.items()}),
              open(R("metrics.json"), "w"), indent=1)
    with open(R("metrics.csv"), "w") as f:
        f.write("model,eval_set,metric,value\n")
        for m in res:
            for s, mm in res[m].items():
                for k, v in mm.items():
                    if isinstance(v, (int, float)) or v is None:
                        f.write(f"{m},{s},{k},{'' if v is None else v}\n")

    # ---------- tables ----------
    main_cols = [("auroc", False), ("tpr_at_1fpr", True), ("tpr_at_5fpr", True), ("f1", False), ("frr_twins", True)]
    lvl_cols = [("auroc", False), ("tpr_at_5fpr", True), ("frr_twins", True)]
    T = ["# Results tables (generated by eval.py; single seed 42)\n",
         "TPR@1%/5% FPR, recall and FRR are percentages. F1, recall and FRR use the threshold chosen on val (max F1). "
         "FRR = share of benign twins flagged. `–` = not defined for that slice.\n"]
    base_models = ["cpmg_full", "cpmg_write_retrieval", "write_only", "content_only", "piguard", "keyword",
                   "tfidf_memory", "tfidf_full", "prov_only"]
    T.append(table("Main results (test, all levels)", res, base_models, ["test_all"], main_cols))
    T.append(table("Per level (test)", res, base_models, ["L1", "L2", "L3"], lvl_cols,
                   "Each level: its positives vs its matched benign twins (L1 also includes the clean ASB, memshield "
                   "and MemoryGraft records). L2 and L3 are derived from AgentPoison records."))
    abl = ["cpmg_full", "abl_no_prov", "abl_no_query", "abl_no_nbrs", "write_only", "cpmg_write_retrieval", "content_only"]
    T.append(table("Ablations (test)", res, abl, ["test_all", "L1", "L2", "L3"], [("auroc", False), ("tpr_at_5fpr", True)]))
    T.append(table("Held-out attack family (BadChain trigger, never in train/val) vs seen family", res, base_models,
                   ["heldout_family", "seen_family_ap"], [("auroc", False), ("tpr_at_5fpr", True), ("frr_twins", True)]))
    T.append(table("Held-out family by level", res, base_models, ["heldout_L1", "heldout_L2", "heldout_L3"],
                   [("auroc", False), ("tpr_at_5fpr", True)]))
    T.append(table("Per source (test)", res, base_models, ["src_AgentPoison", "src_ASB", "src_memshield-bench",
                   "src_MemoryGraft"], [("auroc", False)],
                   "ASB test = 2 held-out agents (all their poisoned records kept). MemoryGraft test is tiny."))
    T.append(table("Cross-source: train without ASB, test on all ASB records", res,
                   ["cpmg_noasb", "tfidf_memory_noasb", "piguard", "keyword"], ["asb_xsource"],
                   [("auroc", False), ("tpr_at_1fpr", True), ("tpr_at_5fpr", True), ("f1", False), ("fpr_all_neg", True)],
                   "Only models whose training never saw ASB are shown. Threshold from each model's own val set."))
    T.append(table("Over-defense on NotInject (339 benign prompts with trigger words)", res, base_models + ["cpmg_noasb"],
                   ["notinject"], [("fpr_all_neg", True)], "False-rejection rate at the val threshold."))
    # efficiency
    eff = ["\n### Cost per check\n", "| Model | Parameters | Latency ms/check (batch 1) | Device | Train time (min) |",
           "|---|---|---|---|---|"]
    for m in ["cpmg_full", "write_only", "content_only", "piguard", "tfidf_memory", "tfidf_full", "keyword", "prov_only"]:
        if m in runs:
            rr = runs[m]
            dev = rr.get("device") or rr.get("gpu", "")
            eff.append(f"| {MODELS[m]} | {rr.get('n_params', 0):,} | {rr.get('latency_ms_bs1', 0):.2f} | {dev} | "
                       f"{(rr['train_seconds']/60):.1f} |" if rr.get("train_seconds") else
                       f"| {MODELS[m]} | {rr.get('n_params', 0):,} | {rr.get('latency_ms_bs1', 0):.2f} | {dev} | – |")
    eff.append("\nCPMG write + retrieval runs both checkpoints, so its cost is the sum of the two rows.")
    T.append("\n".join(eff) + "\n")
    # policy
    pl = ["\n### Authority policy (deterministic; real L1 records; action read from memory text as a proxy)\n",
          "| Set | Label | Records | Proposes protected action | Blocked by policy |", "|---|---|---|---|---|"]
    for name in ("test", "asb_xsource"):
        t = pol[name]["totals"]
        for lab in ("label=1", "label=0"):
            pl.append(f"| {name} | {lab} | {t.get(lab+'|records',0)} | {t.get(lab+'|protected_action',0)} | "
                      f"{t.get(lab+'|blocked',0)} |")
    pl.append("\nPer-source breakdown in `results/policy.json`.")
    T.append("\n".join(pl) + "\n")
    open(R("tables.md"), "w").write("\n".join(T))

    # ---------- plots ----------
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE})
    roc_models = ["cpmg_full", "cpmg_write_retrieval", "content_only", "piguard", "keyword", "tfidf_memory", "tfidf_full"]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    plot_roc(ax, sets["test_all"][1], preds, roc_models, "test", "ROC, test set (all levels)")
    fig.tight_layout()
    fig.savefig(R("plots", "roc_test.png"), dpi=200)
    plt.close(fig)

    lvl_models = ["cpmg_full", "content_only", "tfidf_full", "piguard"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, lvl in zip(axs, ("L1", "L2", "L3")):
        plot_roc(ax, sets[lvl][1], preds, lvl_models, "test", f"ROC, {lvl} (test)")
    fig.tight_layout()
    fig.savefig(R("plots", "roc_by_level.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4))
    style_ax(ax)
    for k, m in enumerate(["cpmg_full", "content_only", "abl_no_prov", "abl_no_query", "abl_no_nbrs", "write_only",
                           "cpmg_noasb"]):
        p = R("logs", f"{m}_train.jsonl")
        if not os.path.exists(p):
            continue
        L = jl(p)
        ax.plot([x["step"] for x in L], [x["loss"] for x in L], color=SERIES[k], linestyle=STYLES[k], linewidth=1.6,
                label=MODELS[m])
    ax.set_title("Training loss (weighted CE, mean over 10 optimiser steps)", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("Optimiser step", fontsize=8, color=INK2)
    ax.set_ylabel("Loss", fontsize=8, color=INK2)
    ax.legend(fontsize=7, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(R("plots", "training_loss.png"), dpi=200)
    plt.close(fig)

    bar_models = ["cpmg_full", "content_only", "piguard", "keyword", "tfidf_memory", "tfidf_full"]
    lv = ["L1", "L2", "L3", "heldout_family"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    style_ax(ax)
    w = 0.8 / len(bar_models)
    for k, m in enumerate(bar_models):
        vals = [res.get(m, {}).get(s, {}).get("auroc", np.nan) for s in lv]
        ax.bar(np.arange(len(lv)) + (k - (len(bar_models) - 1) / 2) * w, vals, width=w * 0.92, color=SERIES[k],
               label=MODELS[m], edgecolor=SURFACE, linewidth=1)
    ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(lv)), ["L1", "L2 (derived)", "L3 (derived)", "Held-out family"], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC (test)", fontsize=8, color=INK2)
    ax.set_title("AUROC by attack level (dashed line = chance)", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=6.5, frameon=False, ncol=3, loc="lower left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(R("plots", "auroc_by_level.png"), dpi=200)
    plt.close(fig)
    print(open(R("tables.md")).read())


if __name__ == "__main__":
    main()
