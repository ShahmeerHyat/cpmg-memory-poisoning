"""Build the methodology + results report from results/ files.

Every number in the report comes from a placeholder resolved here against files produced by our code:
  [[model:eval_set:metric]]      3-decimal value from results/metrics.json
  [[model:eval_set:metric:%]]    percentage (1 decimal)
  [[model:eval_set:auroc_ci95]]  bootstrap 95% CI "[lo, hi]"
  [[stats:split:key]]            from data/processed/stats.json
  [[run:model:key:fmt]]          from results/runs/<model>.json (fmt e.g. .1f, ,)
  [[policy:set:label=1|blocked]] from results/policy.json totals
  [[table:name]]                 generated table
An unresolved placeholder raises an error, so no number can be typed in by hand by accident.

Outputs: report/build/report.md (resolved), report/report.tex (pandoc LaTeX), report/report.pdf (HTML -> WeasyPrint).
Usage: python report/build_report.py
"""
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")

M = json.load(open(os.path.join(RES, "metrics.json")))
RESULTS = M["results"]
STATS = json.load(open(os.path.join(ROOT, "data/processed/stats.json")))
POLICY = json.load(open(os.path.join(RES, "policy.json")))
RUNS = {f[:-5]: json.load(open(os.path.join(RES, "runs", f))) for f in os.listdir(os.path.join(RES, "runs"))
        if f.endswith(".json")}

NAMES = {
    "cpmg_full": "CPMG (retrieval)", "cpmg_write_retrieval": "CPMG write+retrieval", "write_only": "CPMG write-only",
    "abl_no_prov": "– provenance", "abl_no_query": "– query/action", "abl_no_nbrs": "– neighbours",
    "content_only": "Content-only DeBERTa", "piguard": "PIGuard (zero-shot)", "keyword": "Keyword filter",
    "tfidf_memory": "TF-IDF+LR (memory)", "tfidf_full": "TF-IDF+LR (full input)", "prov_only": "Provenance-only LR",
    "cpmg_noasb": "CPMG (trained w/o ASB)", "tfidf_memory_noasb": "TF-IDF+LR memory (w/o ASB)",
}


def g(model, s, metric):
    v = RESULTS.get(model, {}).get(s, {}).get(metric)
    if v is None and metric not in RESULTS.get(model, {}).get(s, {}):
        raise KeyError(f"missing {model}:{s}:{metric}")
    return v


def f3(v):
    return "–" if v is None else f"{v:.3f}"


def pc(v):
    return "–" if v is None else f"{100 * v:.1f}"


def ci(v):
    return "–" if not v or v[0] is None else f"[{v[0]:.3f}, {v[1]:.3f}]"


def md_table(header, rows, align=None):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align or ["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def opt(model, s, metric, fn):
    try:
        return fn(g(model, s, metric))
    except KeyError:
        return "–"


def t_data():
    rows = []
    for sp in ("train", "val", "test", "test_asb_xsource", "test_notinject"):
        st = STATS[sp]
        lv = st["by_level_label"]
        get = lambda k: lv.get(k, 0)
        rows.append([sp.replace("_", " "), f"{st['n']:,}", f"{st['pos']:,}", f"{st['neg']:,}", f"{st['groups']:,}",
                     f"{get('L1|1')}/{get('L1|0')}", f"{get('L2|1')}/{get('L2|0')}", f"{get('L3|1')}/{get('L3|0')}",
                     get("bg|0")])
    return md_table(["Split", "Examples", "Poisoned", "Clean", "Groups", "L1 pos/neg", "L2 pos/neg", "L3 pos/neg",
                     "Background clean"], rows)


def t_main():
    rows = []
    for m in ["cpmg_full", "cpmg_write_retrieval", "write_only", "content_only", "piguard", "keyword", "tfidf_memory",
              "tfidf_full", "prov_only"]:
        if m not in RESULTS:
            continue
        s = "test_all"
        rows.append([NAMES[m], opt(m, s, "auroc", f3), opt(m, s, "auroc_ci95", ci), opt(m, s, "tpr_at_1fpr", pc),
                     opt(m, s, "tpr_at_5fpr", pc), opt(m, s, "f1", f3), opt(m, s, "frr_twins", pc),
                     opt(m, "notinject", "fpr_all_neg", pc)])
    return md_table(["Model", "AUROC", "95% CI", "TPR@1%", "TPR@5%", "F1", "FRR twins %", "FRR NotInject %"], rows)


def t_levels():
    rows = []
    for m in ["cpmg_full", "cpmg_write_retrieval", "write_only", "content_only", "piguard", "keyword", "tfidf_memory",
              "tfidf_full"]:
        if m not in RESULTS:
            continue
        r = [NAMES[m]]
        for lv in ("L1", "L2", "L3"):
            r += [opt(m, lv, "auroc", f3), opt(m, lv, "tpr_at_5fpr", pc), opt(m, lv, "frr_twins", pc)]
        rows.append(r)
    return md_table(["Model", "L1 AUROC", "L1 TPR@5%", "L1 FRR%", "L2 AUROC", "L2 TPR@5%", "L2 FRR%", "L3 AUROC",
                     "L3 TPR@5%", "L3 FRR%"], rows)


def t_ablation():
    rows = []
    for m in ["cpmg_full", "abl_no_prov", "abl_no_query", "abl_no_nbrs", "write_only", "cpmg_write_retrieval",
              "content_only"]:
        if m not in RESULTS:
            continue
        r = [NAMES[m]]
        for s in ("test_all", "L1", "L2", "L3", "heldout_family"):
            r.append(opt(m, s, "auroc", f3))
        r.append(opt(m, "test_all", "tpr_at_5fpr", pc))
        rows.append(r)
    return md_table(["Variant", "All", "L1", "L2", "L3", "Held-out fam.", "TPR@5% (all)"], rows)


def t_generalisation():
    rows = []
    for m in ["cpmg_full", "write_only", "content_only", "piguard", "keyword", "tfidf_memory", "tfidf_full"]:
        if m not in RESULTS:
            continue
        rows.append([NAMES[m], opt(m, "heldout_family", "auroc", f3), opt(m, "seen_family_ap", "auroc", f3),
                     opt(m, "heldout_L1", "auroc", f3), opt(m, "heldout_L2", "auroc", f3),
                     opt(m, "heldout_L3", "auroc", f3), opt(m, "seen_L3", "auroc", f3)])
    return md_table(["Model", "Held-out family", "Seen family", "Held-out L1", "Held-out L2", "Held-out L3",
                     "Seen L3"], rows)


def t_xsource():
    rows = []
    for m in ["cpmg_noasb", "tfidf_memory_noasb", "piguard", "keyword"]:
        if m not in RESULTS:
            continue
        s = "asb_xsource"
        rows.append([NAMES[m], opt(m, s, "auroc", f3), opt(m, s, "auroc_ci95", ci), opt(m, s, "tpr_at_5fpr", pc),
                     opt(m, s, "recall", pc), opt(m, s, "fpr_all_neg", pc)])
    return md_table(["Model (never trained on ASB)", "AUROC", "95% CI", "TPR@5%", "Recall @ val thr. %",
                     "FPR @ val thr. %"], rows)


def t_cost():
    rows = []
    for m in ["cpmg_full", "write_only", "content_only", "piguard", "tfidf_memory", "tfidf_full", "keyword"]:
        r = RUNS.get(m)
        if not r:
            continue
        rows.append([NAMES[m], f"{r.get('n_params', 0):,}", f"{r['latency_ms_bs1']:.2f}",
                     "GPU" if m in ("cpmg_full", "write_only", "content_only", "piguard") else "CPU",
                     f"{r['train_seconds'] / 60:.1f}" if r.get("train_seconds") else "–"])
    return md_table(["Model", "Parameters / features", "ms per check (batch 1)", "Device", "Training (min)"], rows)


def t_policy():
    rows = []
    for name, label in (("test", "Test (real L1 records)"), ("asb_xsource", "All ASB records")):
        t = POLICY[name]["totals"]
        for lab, nm in (("label=1", "poisoned"), ("label=0", "clean")):
            rows.append([label, nm, t.get(f"{lab}|records", 0), t.get(f"{lab}|protected_action", 0),
                         t.get(f"{lab}|blocked", 0)])
    return md_table(["Set", "Records", "Count", "Propose a protected action", "Blocked by policy"], rows)


TABLES = dict(data=t_data, main=t_main, levels=t_levels, ablation=t_ablation, generalisation=t_generalisation,
              xsource=t_xsource, cost=t_cost, policy=t_policy)


def resolve(tok):
    parts = tok.split(":")
    if parts[0] == "table":
        return TABLES[parts[1]]()
    if parts[0] == "stats":
        v = STATS[parts[1]]
        for k in parts[2:]:
            v = v[k]
        return f"{v:,}" if isinstance(v, int) else str(v)
    if parts[0] == "run":
        v = RUNS[parts[1]]
        rest = parts[2:]
        while rest and isinstance(v, dict):
            v = v[rest.pop(0)]
        return format(v, rest[0]) if rest else str(v)
    if parts[0] == "policy":
        return str(POLICY[parts[1]]["totals"].get(parts[2], 0))
    model, s, metric = parts[:3]
    v = g(model, s, metric)
    if metric == "auroc_ci95":
        return ci(v)
    if len(parts) > 3 and parts[3] == "%":
        return pc(v)
    return f3(v)


def main():
    src = open(os.path.join(HERE, "report.md")).read()
    out = re.sub(r"\[\[([^\]]+)\]\]", lambda m: resolve(m.group(1)), src)
    os.makedirs(os.path.join(HERE, "build"), exist_ok=True)
    resolved = os.path.join(HERE, "build", "report.md")
    open(resolved, "w").write(out)
    # LaTeX source (for later compilation with the ACL template) and HTML -> PDF
    subprocess.run(["pandoc", resolved, "-s", "-o", os.path.join(HERE, "report.tex"), "--resource-path", HERE],
                   check=True)
    html = os.path.join(HERE, "build", "report.html")
    subprocess.run(["pandoc", resolved, "-s", "-o", html, "--css", os.path.join(HERE, "report.css"),
                    "--resource-path", HERE, "--embed-resources", "--metadata", "pagetitle=CPMG report"], check=True)
    from weasyprint import HTML
    HTML(html, base_url=HERE).write_pdf(os.path.join(HERE, "report.pdf"))
    print("wrote report/report.pdf, report/report.tex")


if __name__ == "__main__":
    main()
