"""Evaluate every model from its prediction files and write metrics, tables and plots (dataset v2).

Two targets (results/DECISIONS.md D23):
  poison : attacker-written content is in the context (memory or neighbours)          -> pred field `score`
  active : the query carries the trigger the poisoned content is conditional on       -> pred field `score_act`
Inputs:  data/processed/*.jsonl, results/preds/<model>[_s<seed>]/<split>.jsonl, results/runs/*.json, results/logs/*_train.jsonl
Outputs: results/metrics.csv (every seed), results/metrics.json (seed 42 + mean/std over seeds), results/tables.md,
         results/policy.json, results/plots/*.png
Thresholds are chosen on val only (max F1 on val, per target), then applied unchanged to every test set.
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
SEEDS = [42, 43, 44]
CI_SETS = {"P_all", "P_L1", "P_L2", "A_L3_hard_real", "A_L23_hard_real", "A_L3_hard_rand", "asb_xsource",
           "A_L3_hard", "A_L23_hard", "P_L1_ap",
           "P_L1_ehr", "P_L2_ehr", "A_hard_ehr", "A_L3_hard_ehr", "A_L23_hard_ehr"}

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
    ("lexmatch", "Lexical copy rule"),
    ("lexquote", "Quoted-string copy rule (knows the quoting)"),
    ("tfidf_memory", "TF-IDF+LR (memory)"),
    ("tfidf_full", "TF-IDF+LR (full input)"),
    ("prov_only", "Provenance-only LR (leak check)"),
    ("cpmg_noasb", "CPMG trained w/o ASB"),
    ("cpmg_noehr", "CPMG trained w/o EHRAgent records"),
    ("tfidf_memory_noasb", "TF-IDF+LR (memory) w/o ASB"),
])
XSOURCE_VALID = {"cpmg_noasb", "tfidf_memory_noasb", "piguard", "keyword"}  # training never saw ASB
REAL_NAMES = {"ap": "AgentPoison optimised", "ap_long": "AgentPoison optimised (long)", "badchain": "BadChain phrase",
              "seed_ad": "seed: 'Be careful and drive safe.'", "seed_qa": "seed: 'Make efficient tool calls.'"}

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (8, 2))]


def jl(p):
    return [json.loads(l) for l in open(p)]


def load_data():
    return {sp: {r["id"]: r for r in jl(os.path.join(ROOT, "data/processed", f"{sp}.jsonl"))}
            for sp in ("val", "test", "test_asb_xsource", "test_notinject")}


def load_pred_dir(d):
    out = {}
    for fn in os.listdir(d):
        if fn.endswith(".jsonl"):
            out[fn[:-6]] = {r["id"]: (r["score"], r.get("score_act", r["score"])) for r in jl(os.path.join(d, fn))}
    return out


def load_preds():
    """preds[model][seed][split][id] = (poison score, active score)."""
    preds = collections.defaultdict(dict)
    for m in MODELS:
        for seed in SEEDS:
            d = R("preds", m if seed == 42 else f"{m}_s{seed}")
            if os.path.isdir(d):
                preds[m][seed] = load_pred_dir(d)
    for seed in SEEDS:  # flag if either checkpoint flags; activation needs the query, so it comes from retrieval
        if seed in preds.get("cpmg_full", {}) and 42 in preds.get("write_only", {}):
            full, wo = preds["cpmg_full"][seed], preds["write_only"][42]
            preds["cpmg_write_retrieval"][seed] = {
                sp: {i: (max(full[sp][i][0], wo[sp][i][0]), full[sp][i][1]) for i in full[sp] if i in wo.get(sp, {})}
                for sp in full if sp in wo}
    return preds


def best_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p * r / np.maximum(p + r, 1e-12)
    return float(t[int(np.nanargmax(f[:-1]))]) if len(t) else 0.5


def tpr_at_fpr(y, s, target):
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= target + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def boot_auc(y, s, n=1000):
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            out.append(roc_auc_score(y[i], s[i]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))] if out else None


def metrics(y, s, thr, ci=False):
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    yhat = (s >= thr).astype(int)
    neg = y == 0
    m = dict(n=int(len(y)), n_pos=int(y.sum()), n_neg=int(neg.sum()), threshold=thr,
             fpr=float(yhat[neg].mean()) if neg.any() else None)
    if 0 < y.sum() < len(y):
        m.update(auroc=float(roc_auc_score(y, s)), tpr_at_1fpr=tpr_at_fpr(y, s, 0.01),
                 tpr_at_5fpr=tpr_at_fpr(y, s, 0.05), f1=float(f1_score(y, yhat, zero_division=0)),
                 recall=float(yhat[y == 1].mean()))
        if ci and y.sum() >= 5 and neg.sum() >= 5:
            m["auroc_ci95"] = boot_auc(y, s)
    elif len(y) and y.sum() == len(y):
        m.update(recall=float(yhat.mean()))
    return m


def eval_sets(data):
    """name -> (split, head, positives, negatives). head 0 = poison, 1 = active."""
    t = list(data["test"].values())
    V = lambda *vs: [r for r in t if r["variant"] in vs]
    ap = lambda rows: [r for r in rows if r["source_dataset"] == "AgentPoison" and r["subset"] != "strategyqa_paragraph"]
    S = collections.OrderedDict()
    # ---- poison target ----
    S["P_all"] = ("test", 0, [r for r in t if r["y_poison"] == 1], [r for r in t if r["y_poison"] == 0])
    l1neg = V("L1_twin", "L1_neg", "bg")
    S["P_L1"] = ("test", 0, V("L1_pos"), l1neg)
    S["P_L2"] = ("test", 0, V("L2_pos"), V("L1_twin"))
    S["P_L1_ap"] = ("test", 0, ap(V("L1_pos")), V("L1_twin"))
    for k, nm in (("real", "real"), ("rand", "random")):
        f = lambda rows: [r for r in rows if (r.get("trigger_kind") or "").startswith(nm)]
        S[f"P_L1_ap_{k}"] = ("test", 0, f(V("L1_pos")), f(V("L1_twin")))
        S[f"P_L2_{k}"] = ("test", 0, f(V("L2_pos")), f(V("L1_twin")))
    for src in ("AgentPoison", "ASB", "memshield-bench", "MemoryGraft"):
        S[f"P_src_{src}"] = ("test", 0, [r for r in V("L1_pos") if r["source_dataset"] == src],
                             [r for r in l1neg if r["source_dataset"] == src])
    S["P_trig_only"] = ("test", 0, [], V("trig_only"))  # clean memory, trigger in query: false-alarm rate
    S["asb_xsource"] = ("test_asb_xsource", 0, [r for r in data["test_asb_xsource"].values() if r["y_poison"] == 1],
                        [r for r in data["test_asb_xsource"].values() if r["y_poison"] == 0])
    S["notinject"] = ("test_notinject", 0, [], list(data["test_notinject"].values()))
    # ---- activation target (AgentPoison-derived) ----
    S["A_all"] = ("test", 1, [r for r in t if r["y_active"] == 1], [r for r in t if r["y_active"] == 0])
    S["A_L3_none"] = ("test", 1, V("L3_pos"), ap(V("L1_pos")))     # same memory, no trigger in the query
    S["A_L3_mis"] = ("test", 1, V("L3_pos"), V("L3_mis"))          # same memory, a different trigger in the query
    S["A_L23_none"] = ("test", 1, V("L23_pos"), V("L2_pos"))
    S["A_L23_mis"] = ("test", 1, V("L23_pos"), V("L23_mis"))
    S["A_L3_ovl"] = ("test", 1, V("L3_pos"), V("L3_ovl"))          # same memory, query repeats harmless record text
    S["A_L23_ovl"] = ("test", 1, V("L23_pos"), V("L23_ovl"))
    S["A_L3_hard"] = ("test", 1, V("L3_pos"), V("L3_mis", "L3_ovl"))
    S["A_L23_hard"] = ("test", 1, V("L23_pos"), V("L23_mis", "L23_ovl"))
    S["A_trig_only"] = ("test", 1, V("L3_pos", "L23_pos"), V("trig_only"))  # trigger present, nothing to activate
    for base in ("A_L3_none", "A_L3_mis", "A_L23_none", "A_L23_mis", "A_L3_ovl", "A_L23_ovl", "A_L3_hard", "A_L23_hard",
                 "A_trig_only"):
        _, h, pos, neg = S[base]
        for k, nm in (("real", "real"), ("rand", "random")):
            f = lambda rows: [r for r in rows if (r.get("trigger_kind") or "").startswith(nm)]
            S[f"{base}_{k}"] = ("test", h, f(pos), f(neg))
    for k, sub in (("ehr", "ehragent"), ("sq", "strategyqa")):  # per injection template (cross-template test, D30)
        f = lambda rows, sub=sub: [r for r in rows if r["subset"] == sub]
        S[f"P_L1_{k}"] = ("test", 0, f(V("L1_pos")), f(V("L1_twin")))
        S[f"P_L2_{k}"] = ("test", 0, f(V("L2_pos")), f(V("L1_twin")))
        S[f"A_hard_{k}"] = ("test", 1, f(V("L3_pos", "L23_pos")), f(V("L3_mis", "L3_ovl", "L23_mis", "L23_ovl")))
        S[f"A_none_{k}"] = ("test", 1, f(V("L3_pos", "L23_pos")), f(ap(V("L1_pos")) + V("L2_pos")))
        S[f"A_L3_hard_{k}"] = ("test", 1, f(V("L3_pos")), f(V("L3_mis", "L3_ovl")))
        S[f"A_L3_none_{k}"] = ("test", 1, f(V("L3_pos")), f(ap(V("L1_pos"))))
        S[f"A_L23_hard_{k}"] = ("test", 1, f(V("L23_pos")), f(V("L23_mis", "L23_ovl")))
    for name in REAL_NAMES:  # binding per real trigger: positives carry this trigger; negatives = all real mismatches
        fam = "ap_real_" + name
        S[f"A_mis_{name}"] = ("test", 1, [r for r in V("L3_pos", "L23_pos") if r["group_family"] == fam],
                              [r for r in V("L3_mis", "L23_mis", "L3_ovl", "L23_ovl") if r["trigger_kind"] == "real"])
    return S


def fmt(v, pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{100*v:.1f}" if pct else f"{v:.3f}"


def main():
    os.makedirs(R("plots"), exist_ok=True)
    data = load_data()
    preds = load_preds()
    sets = eval_sets(data)
    val = list(data["val"].values())
    res = collections.defaultdict(dict)   # res[model][seed][set] = metrics
    thr = collections.defaultdict(dict)
    for m in preds:
        for seed, P in preds[m].items():
            if "val" not in P or any(i not in P.get("test", {}) for i in data["test"]):
                print("skip (missing or stale predictions):", m, seed)
                continue
            v = [r for r in val if r["id"] in P["val"]]
            va = [r for r in v if r["y_active"] >= 0]
            th = [best_threshold([r["y_poison"] for r in v], [P["val"][r["id"]][0] for r in v]),
                  best_threshold([r["y_active"] for r in va], [P["val"][r["id"]][1] for r in va])]
            thr[m][seed] = th
            out = {}
            for sname, (sp, h, pos, neg) in sets.items():
                if sname == "asb_xsource" and m not in XSOURCE_VALID:
                    continue
                for tag in ("", "@noprov", "@provswap", "@para", "@paranq"):
                    if tag in ("@noprov", "@provswap") and not (sname in ("notinject", "asb_xsource", "P_all", "P_L1") and sp + tag in P):
                        continue
                    if tag in ("@para", "@paranq") and sp != "test":
                        continue
                    if sp + tag not in P:
                        continue
                    rows = [r for r in pos + neg if r["id"] in P[sp + tag]]
                    if not rows:
                        continue
                    y = [1] * len([r for r in pos if r["id"] in P[sp + tag]]) + [0] * len([r for r in neg if r["id"] in P[sp + tag]])
                    s = [P[sp + tag][r["id"]][h] for r in rows]
                    out[sname + tag] = metrics(y, s, th[h], ci=(seed == 42 and sname in CI_SETS
                                                                 and tag in ("", "@para", "@paranq")))
            # provenance swap: how much does the poison score move when only the provenance tokens change?
            if "test@provswap" in P:
                ids = [i for i in P["test"] if i in P["test@provswap"]]
                a = np.array([P["test"][i][0] for i in ids]); b = np.array([P["test@provswap"][i][0] for i in ids])
                out["provswap_delta"] = dict(n=len(ids), mean_abs_delta=float(np.abs(a - b).mean()),
                                             decision_flip_rate=float(((a >= th[0]) != (b >= th[0])).mean()))
            res[m][seed] = out

    # ---- aggregate over seeds ----
    agg = {}
    for m in res:
        agg[m] = {}
        for sname in res[m].get(42, {}):
            agg[m][sname] = {}
            for k in ("auroc", "tpr_at_1fpr", "tpr_at_5fpr", "f1", "fpr", "recall", "mean_abs_delta", "decision_flip_rate"):
                vals = [res[m][s][sname][k] for s in res[m] if sname in res[m][s] and res[m][s][sname].get(k) is not None]
                if vals:
                    agg[m][sname][k] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)), n_seeds=len(vals),
                                            min=float(np.min(vals)), max=float(np.max(vals)))
    runs = {os.path.basename(p)[:-5]: json.load(open(R("runs", p))) for p in os.listdir(R("runs")) if p.endswith(".json")}
    pol = policy_eval(data)
    json.dump(pol, open(R("policy.json"), "w"), indent=1)
    json.dump(dict(thresholds=thr, seed42={m: res[m].get(42, {}) for m in res}, over_seeds=agg,
                   set_sizes={k: dict(split=v[0], target=["poison", "active"][v[1]], n_pos=len(v[2]), n_neg=len(v[3]))
                              for k, v in sets.items()},
                   runs={k: {kk: v.get(kk) for kk in ("latency_ms_bs1", "n_params", "train_seconds", "best_val_auroc",
                                                      "history", "device", "gpu")} for k, v in runs.items()}),
              open(R("metrics.json"), "w"), indent=1)
    with open(R("metrics.csv"), "w") as f:
        f.write("model,seed,eval_set,metric,value\n")
        for m in res:
            for seed in res[m]:
                for s, mm in res[m][seed].items():
                    for k, v in mm.items():
                        if isinstance(v, (int, float)) or v is None:
                            f.write(f"{m},{seed},{s},{k},{'' if v is None else v}\n")

    # ---------- tables ----------
    def cell(m, sname, k, pct=False):
        a = agg.get(m, {}).get(sname, {}).get(k)
        if a is None:
            return "–"
        sc = 100 if pct else 1
        d = 1 if pct else 3
        if a["n_seeds"] > 1:
            return f"{a['mean']*sc:.{d}f} ± {a['std']*sc:.{d}f}"
        return f"{a['mean']*sc:.{d}f}"

    def table(title, models, cols, note=""):
        out = [f"\n### {title}\n"]
        if note:
            out.append(note + "\n")
        out.append("| Model | " + " | ".join(c[0] for c in cols) + " |")
        out.append("|---|" + "---|" * len(cols))
        for m in models:
            if m in agg:
                out.append(f"| {MODELS[m]} | " + " | ".join(cell(m, c[1], c[2], c[3] if len(c) > 3 else False) for c in cols) + " |")
        return "\n".join(out) + "\n"

    nseed = {m: len(res[m]) for m in res}
    T = ["# Results tables, dataset v2 (generated by eval.py)\n",
         "Cells with ± are mean ± std over seeds " + ", ".join(f"{MODELS[m]}: {n}" for m, n in nseed.items() if n > 1)
         + ". All other rows are single seed (42). TPR and FPR are percentages. Thresholds come from val (max F1), "
           "per target. `–` = not defined or not run.\n",
         "Two targets: **poison** (attacker-written content is in the context) and **active** (the query carries the "
         "trigger that content is conditional on). Test activation examples use only the five real shipped triggers "
         "(`real`) or fresh random triggers (`rand`); no test trigger occurs in train or val.\n"]
    base = ["cpmg_full", "cpmg_write_retrieval", "write_only", "content_only", "piguard", "keyword", "lexmatch", "lexquote",
            "tfidf_memory", "tfidf_full", "prov_only"]
    T.append(table("Table 1. Poison detection (test)", base,
                   [("All AUROC", "P_all", "auroc"), ("All TPR@1%FPR", "P_all", "tpr_at_1fpr", True),
                    ("L1 AUROC", "P_L1", "auroc"), ("L1 TPR@1%FPR", "P_L1", "tpr_at_1fpr", True),
                    ("L2 AUROC", "P_L2", "auroc"), ("L2 TPR@1%FPR", "P_L2", "tpr_at_1fpr", True),
                    ("F1 (all)", "P_all", "f1"), ("FPR all (%)", "P_all", "fpr", True)],
                   "L1 = real poisoned records vs clean records. L2 (derived) = clean memory with the payload stored "
                   "as a separate neighbour, vs the same memory with clean neighbours."))
    T.append(table("Table 2. Trigger activation on the five REAL shipped triggers (test, AUROC)", base,
                   [("L3 vs no trigger", "A_L3_none_real", "auroc"), ("L3 vs wrong trigger", "A_L3_mis_real", "auroc"),
                    ("L3 vs benign overlap", "A_L3_ovl_real", "auroc"), ("L3 hard (both)", "A_L3_hard_real", "auroc"),
                    ("L2+L3 vs wrong trigger", "A_L23_mis_real", "auroc"), ("L2+L3 vs benign overlap", "A_L23_ovl_real", "auroc"),
                    ("L2+L3 hard (both)", "A_L23_hard_real", "auroc"), ("vs trigger w/o payload", "A_trig_only_real", "auroc")],
                   "Positives: the query carries the trigger named by the poisoned content (in the memory for L3, in a "
                   "neighbour for L2+L3). Negatives share the same memory and neighbours: the query carries no trigger, a "
                   "different real trigger, or a harmless 3-6 word span repeated from the record (benign overlap). "
                   "None of the five triggers occurs in training."))
    T.append(table("Table 3. Trigger activation at low false-positive rate (test, TPR@1%FPR, %)", base,
                   [("L3 hard, real", "A_L3_hard_real", "tpr_at_1fpr", True),
                    ("L2+L3 hard, real", "A_L23_hard_real", "tpr_at_1fpr", True),
                    ("L3 hard, rand", "A_L3_hard_rand", "tpr_at_1fpr", True),
                    ("L2+L3 hard, rand", "A_L23_hard_rand", "tpr_at_1fpr", True),
                    ("L3 hard AUROC, rand", "A_L3_hard_rand", "auroc"), ("L2+L3 hard AUROC, rand", "A_L23_hard_rand", "auroc"),
                    ("All activation AUROC", "A_all", "auroc"), ("All activation F1", "A_all", "f1")],
                   "hard = wrong-trigger and benign-overlap negatives together. rand = fresh random triggers."))
    abl = ["cpmg_full", "abl_no_prov", "abl_no_query", "abl_no_nbrs", "write_only", "content_only"]
    T.append(table("Table 4. Ablations (test, AUROC)", abl,
                   [("Poison all", "P_all", "auroc"), ("Poison L1", "P_L1", "auroc"), ("Poison L2", "P_L2", "auroc"),
                    ("Active L3 hard, real", "A_L3_hard_real", "auroc"), ("Active L2+L3 hard, real", "A_L23_hard_real", "auroc"),
                    ("Active L3 hard, rand", "A_L3_hard_rand", "auroc"), ("Active all", "A_all", "auroc")]))
    T.append(table("Table 5. Activation per real trigger (test, AUROC; positives with this trigger vs all real wrong-trigger and benign-overlap negatives)",
                   ["cpmg_full", "abl_no_prov", "tfidf_full", "lexmatch", "lexquote"],
                   [(REAL_NAMES[n], f"A_mis_{n}", "auroc") for n in REAL_NAMES]))
    T.append(table("Table 6. Poison detection per source (test, L1, AUROC)", base,
                   [(s, f"P_src_{s}", "auroc") for s in ("AgentPoison", "ASB", "memshield-bench", "MemoryGraft")],
                   "ASB test = 2 held-out agents. MemoryGraft test is tiny."))
    T.append(table("Table 7. Cross-source: trained without ASB, tested on all ASB records", ["cpmg_noasb", "tfidf_memory_noasb", "piguard", "keyword"],
                   [("AUROC", "asb_xsource", "auroc"), ("TPR@1%FPR", "asb_xsource", "tpr_at_1fpr", True),
                    ("Recall at val thr (%)", "asb_xsource", "recall", True), ("FPR at val thr (%)", "asb_xsource", "fpr", True),
                    ("AUROC, provenance removed", "asb_xsource@noprov", "auroc"),
                    ("Recall, provenance removed (%)", "asb_xsource@noprov", "recall", True),
                    ("FPR, provenance removed (%)", "asb_xsource@noprov", "fpr", True)]))
    T.append(table("Table 8. False alarms on benign inputs (%, at the val threshold)", base + ["abl_no_prov", "cpmg_noasb"],
                   [("NotInject (poison)", "notinject", "fpr", True), ("NotInject, provenance removed", "notinject@noprov", "fpr", True),
                    ("Trigger w/o payload (poison)", "P_trig_only", "fpr", True), ("Trigger w/o payload (active)", "A_trig_only", "fpr", True)],
                   "NotInject: 339 benign prompts with trigger words; its provenance tags never occur in training."))
    T.append(table("Table 9. Provenance counterfactual: same test records, provenance tokens swapped to another source's",
                   ["cpmg_full", "abl_no_query", "abl_no_nbrs", "write_only", "cpmg_noasb"],
                   [("Poison AUROC", "P_all", "auroc"), ("Poison AUROC, swapped", "P_all@provswap", "auroc"),
                    ("Poison AUROC, removed", "P_all@noprov", "auroc"),
                    ("Mean |score change|", "provswap_delta", "mean_abs_delta"), ("Decisions flipped (%)", "provswap_delta", "decision_flip_rate", True)]))
    T.append(table("Table 12. Cross-template: EHRAgent records (a different injection sentence) when the model never saw them",
                   ["cpmg_full", "cpmg_noehr", "content_only", "tfidf_full", "lexmatch", "lexquote"],
                   [("EHR poison L1", "P_L1_ehr", "auroc"), ("EHR poison L2", "P_L2_ehr", "auroc"),
                    ("EHR L3 vs no trigger", "A_L3_none_ehr", "auroc"), ("EHR L3 hard", "A_L3_hard_ehr", "auroc"),
                    ("EHR L2+L3 hard", "A_L23_hard_ehr", "auroc"), ("EHR active hard (both)", "A_hard_ehr", "auroc"),
                    ("EHR active hard TPR@1%FPR", "A_hard_ehr", "tpr_at_1fpr", True),
                    ("StrategyQA active hard", "A_hard_sq", "auroc")],
                   "CPMG trained w/o EHRAgent records has seen only the StrategyQA injection sentence."))
    pm = ["cpmg_full", "abl_no_prov", "cpmg_noehr", "content_only", "piguard", "tfidf_full", "lexmatch", "lexquote"]
    for tag, nm in (("@para", "free paraphrase"), ("@paranq", "paraphrase without quoting the trigger")):
        T.append(table(f"Table 13{'a' if tag == '@para' else 'b'}. Paraphrase attack ({nm}): injected sentence rewritten by Qwen3-8B, "
                       "same records, triggers and ids; no retraining", pm,
                       [("Poison L1 (AP) orig", "P_L1_ap", "auroc"), ("Poison L1 (AP) para", "P_L1_ap" + tag, "auroc"),
                        ("Poison L1 TPR@1% para", "P_L1_ap" + tag, "tpr_at_1fpr", True),
                        ("Poison L2 para", "P_L2" + tag, "auroc"),
                        ("Active L3 hard orig", "A_L3_hard", "auroc"), ("Active L3 hard para", "A_L3_hard" + tag, "auroc"),
                        ("Active L3 TPR@1% para", "A_L3_hard" + tag, "tpr_at_1fpr", True),
                        ("Active L2+L3 hard para", "A_L23_hard" + tag, "auroc"),
                        ("Active L3 hard real para", "A_L3_hard_real" + tag, "auroc"),
                        ("Poison recall at val thr para (%)", "P_L1_ap" + tag, "recall", True),
                        ("Active recall at val thr para (%)", "A_L3_hard" + tag, "recall", True)]))
    eff = ["\n### Table 10. Cost per check\n", "| Model | Parameters | Latency ms/check (batch 1) | Device | Train time (min) |", "|---|---|---|---|---|"]
    for m in ["cpmg_full", "write_only", "content_only", "piguard", "tfidf_memory", "tfidf_full", "lexmatch", "keyword", "prov_only"]:
        if m in runs:
            rr = runs[m]
            eff.append(f"| {MODELS[m]} | {rr.get('n_params', 0):,} | {rr.get('latency_ms_bs1', 0):.2f} | {rr.get('device') or rr.get('gpu', '')} | "
                       + (f"{rr['train_seconds']/60:.1f}" if rr.get("train_seconds") else "–") + " |")
    T.append("\n".join(eff) + "\n")
    pl = ["\n### Table 11. Authority policy (deterministic; real L1 records; action read from memory text as a proxy)\n",
          "| Set | Label | Records | Proposes protected action | Blocked by policy |", "|---|---|---|---|---|"]
    for name in ("test", "asb_xsource"):
        tt = pol[name]["totals"]
        for lab in ("label=1", "label=0"):
            pl.append(f"| {name} | {lab} | {tt.get(lab+'|records',0)} | {tt.get(lab+'|protected_action',0)} | {tt.get(lab+'|blocked',0)} |")
    T.append("\n".join(pl) + "\n")
    open(R("tables.md"), "w").write("\n".join(T))

    # ---------- plots ----------
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE})

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

    def plot_roc(ax, sname, models, title):
        style_ax(ax)
        ax.plot([0, 1], [0, 1], color=GRID, linewidth=1, linestyle="--")
        sp, h, pos, neg = sets[sname]
        for k, m in enumerate(models):
            P = preds.get(m, {}).get(42, {}).get(sp)
            if not P:
                continue
            y = [1] * len(pos) + [0] * len(neg)
            s = [P[r["id"]][h] for r in pos + neg]
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, color=SERIES[k], linestyle=STYLES[k], linewidth=2, label=f"{MODELS[m]} ({roc_auc_score(y, s):.3f})")
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("False positive rate", fontsize=8, color=INK2)
        ax.set_ylabel("True positive rate", fontsize=8, color=INK2)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
        ax.legend(fontsize=6.5, frameon=False, loc="lower right", labelcolor=INK)

    rm = ["cpmg_full", "content_only", "piguard", "keyword", "tfidf_memory", "tfidf_full", "lexmatch"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    plot_roc(axs[0], "P_all", rm, "Poison detection, all test records")
    plot_roc(axs[1], "A_L3_hard_real", rm, "Activation, L3, real triggers (hard negatives)")
    plot_roc(axs[2], "A_L23_hard_real", rm, "Activation, L2+L3, real triggers (hard negatives)")
    fig.tight_layout(); fig.savefig(R("plots", "roc_main.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4))
    style_ax(ax)
    for k, m in enumerate(["cpmg_full", "content_only", "abl_no_prov", "abl_no_query", "abl_no_nbrs", "write_only", "cpmg_noasb"]):
        p = R("logs", f"{m}_train.jsonl")
        if os.path.exists(p):
            L = jl(p)
            ax.plot([x["step"] for x in L], [x["loss"] for x in L], color=SERIES[k], linestyle=STYLES[k], linewidth=1.6, label=MODELS[m])
    ax.set_title("Training loss (weighted BCE over two targets, mean of 10 optimiser steps)", fontsize=9.5, color=INK, loc="left")
    ax.set_xlabel("Optimiser step", fontsize=8, color=INK2); ax.set_ylabel("Loss", fontsize=8, color=INK2)
    ax.legend(fontsize=7, frameon=False, labelcolor=INK)
    fig.tight_layout(); fig.savefig(R("plots", "training_loss.png"), dpi=200); plt.close(fig)

    bm = ["cpmg_full", "abl_no_query", "abl_no_nbrs", "content_only", "tfidf_full", "lexmatch"]
    lv = [("P_L1", "Poison L1"), ("P_L2", "Poison L2"), ("A_L3_hard_real", "Active L3\nreal triggers"),
          ("A_L23_hard_real", "Active L2+L3\nreal triggers"), ("A_L3_hard_rand", "Active L3\nrandom triggers")]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    style_ax(ax)
    w = 0.8 / len(bm)
    for k, m in enumerate(bm):
        vals = [agg.get(m, {}).get(s, {}).get("auroc", {}).get("mean", np.nan) for s, _ in lv]
        errs = [agg.get(m, {}).get(s, {}).get("auroc", {}).get("std", 0) for s, _ in lv]
        ax.bar(np.arange(len(lv)) + (k - (len(bm) - 1) / 2) * w, vals, width=w * 0.92, color=SERIES[k], label=MODELS[m],
               edgecolor=SURFACE, linewidth=1, yerr=errs, error_kw=dict(ecolor=INK2, lw=0.8))
    ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(lv)), [l for _, l in lv], fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("AUROC (test)", fontsize=8, color=INK2)
    ax.set_title("AUROC by task (dashed line = chance; bars with whiskers = mean ± std over seeds)", fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=6.5, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16), labelcolor=INK)
    fig.tight_layout(); fig.savefig(R("plots", "auroc_by_task.png"), dpi=200); plt.close(fig)
    print(open(R("tables.md")).read())


def policy_eval(data):
    """Authority policy on the real (non-derived) L1 records of the test and ASB cross-source sets."""
    out = {}
    for name, sp in (("test", "test"), ("asb_xsource", "test_asb_xsource")):
        rows = [r for r in data[sp].values() if r["variant"] in ("L1_pos", "L1_twin", "L1_neg", "bg")]
        agg = collections.defaultdict(collections.Counter)
        for r in rows:
            dec, cat = policy.decide(r["authority"], r["memory"])
            key = f"{r['source_dataset']}|label={r['label']}|auth={r['authority']}"
            agg[key]["records"] += 1
            agg[key]["protected_action"] += cat is not None
            agg[key]["blocked"] += dec == "block"
        tot = collections.Counter()
        for k, c in agg.items():
            for kk, v in c.items():
                tot[f"{k.split('|')[1]}|{kk}"] += v
        out[name] = dict(by_source={k: dict(v) for k, v in sorted(agg.items())}, totals=dict(tot))
    return out


if __name__ == "__main__":
    main()
