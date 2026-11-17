"""Build paper/paper.tex from paper/paper.tex.in and compile it. Every number is a placeholder resolved against
files our code wrote, so no number is typed by hand.

  [[m:SET:METRIC]]        mean over seeds, 3 decimals; "a $\\pm$ b" when >1 seed and the std is not ~0   results/metrics.json
  [[m:SET:METRIC:%]]      same as a percentage, 1 decimal
  [[m:SET:METRIC:mean]]   mean only (mean% for a percentage)
  [[ci:m:SET]]            bootstrap 95% CI of the seed-42 AUROC
  [[n:SET:n_pos|n_neg]]   evaluation-set sizes
  [[seeds:m]]             number of seeds evaluated for m
  [[stats:split:key]] / [[stats:key]]   data/processed/stats.json
  [[run:m:key:fmt]]       results/runs/<m>.json  (key may be config.lr)
  [[policy:set:label=1|blocked]]        results/policy.json totals
  [[para:COND:KEY]]       paraphrase audit numbers (data/interim/paraphrases.jsonl)
  [[table:name]]          generated LaTeX table
An unresolved placeholder stops the build.

Usage: python paper/build_paper.py [--review]     (--review also writes the anonymous paper_review.pdf)
Compiles with tectonic or latexmk if installed, else with the local Docker image texlive/texlive:latest-full.
"""
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
M = json.load(open(os.path.join(ROOT, "results/metrics.json")))
AGG, S42, SIZES = M["over_seeds"], M["seed42"], M["set_sizes"]
STATS = json.load(open(os.path.join(ROOT, "data/processed/stats.json")))
POLICY = json.load(open(os.path.join(ROOT, "results/policy.json")))
RUNS = {f[:-5]: json.load(open(os.path.join(ROOT, "results/runs", f)))
        for f in os.listdir(os.path.join(ROOT, "results/runs")) if f.endswith(".json")}

NAMES = {
    "cpmg_full": "CPMG (retrieval)", "cpmg_write_retrieval": "CPMG write+retrieval", "write_only": "CPMG write-time only",
    "abl_no_prov": "CPMG $-$ provenance", "abl_no_query": "CPMG $-$ query", "abl_no_nbrs": "CPMG $-$ neighbours",
    "content_only": "Content-only DeBERTa", "piguard": "PIGuard (zero-shot)", "keyword": "Keyword filter",
    "lexmatch": "Copy rule", "lexquote": "Quoted-copy rule$^\\dagger$",
    "tfidf_memory": "TF-IDF (memory)", "tfidf_full": "TF-IDF (full input)", "prov_only": "Provenance-only LR",
    "cpmg_noasb": "CPMG trained w/o ASB", "tfidf_memory_noasb": "TF-IDF (memory) w/o ASB",
    "cpmg_noehr": "CPMG trained w/o EHRAgent",
}


def para_audit():
    p = os.path.join(ROOT, "data/interim/paraphrases.jsonl")
    if not os.path.exists(p):
        return {}
    rows = [json.loads(l) for l in open(p)]
    rx = re.compile(r"[’'‘]([^’'‘\n]{3,120})[’'‘] in the question")
    out = {}
    for c in ("para", "paranq"):
        ok = [r for r in rows if r[c]]
        out[c] = dict(groups=len(rows), accepted=len(ok), unique=len({r[c] for r in ok}),
                      rule_match=sum(bool(rx.search(r[c].replace("<<T>>", r["trigger"]))) for r in ok))
    return out


PARA = para_audit()


def para_example(cond):
    """A real accepted rewrite (first StrategyQA test record with the BadChain trigger), LaTeX-escaped."""
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data/interim/paraphrases.jsonl"))]
    r = next(r for r in rows if r["subset"] == "strategyqa" and r["trigger"] == "In a parallel universe" and r[cond])
    t = r[cond].replace("<<T>>", "\\textit{" + r["trigger"] + "}")
    for a_, b_ in (("’", "'"), ("‘", "'"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#")):
        t = t.replace(a_, b_)
    return t.replace("'I don't know'", "`I don't know'")



def val(m, s, k, mode=""):
    a = AGG.get(m, {}).get(s, {}).get(k)
    if a is None:
        raise KeyError(f"missing {m}:{s}:{k}")
    pct = "%" in mode
    sc, d = (100, 1) if pct else (1, 3)
    if a["n_seeds"] > 1 and "mean" not in mode and round(a["std"] * sc, d) > 0:
        return f"{a['mean']*sc:.{d}f}\\,$\\pm$\\,{a['std']*sc:.{d}f}"
    return f"{a['mean']*sc:.{d}f}"


def cell(m, s, k, mode=""):
    try:
        return val(m, s, k, mode)
    except KeyError:
        return "--"


def tex_table(models, cols, colspec=None, header=None):
    lines = ["\\begin{tabular}{" + (colspec or ("l" + "r" * len(cols))) + "}", "\\toprule"]
    if header:
        lines.append(header)
    lines += [" & ".join(["Model"] + [c[0] for c in cols]) + " \\\\", "\\midrule"]
    for m in models:
        if m == "---":
            lines.append("\\midrule")
        elif m in AGG:
            # the two rules have no poison score: print a dash instead of their constant output
            cs = ["--" if (m in ("lexmatch", "lexquote") and c[1].startswith("P_")) else
                  cell(m, c[1], c[2], c[3] if len(c) > 3 else "") for c in cols]
            lines.append(" & ".join([NAMES[m]] + cs) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def t_robust():
    """Original vs reworded attack: poison TPR@1% (AgentPoison L1) and activation (L3, hard negatives)."""
    ms = ["cpmg_full", "abl_no_prov", "---", "lexquote", "lexmatch", "tfidf_full", "content_only", "piguard"]
    tags = [("", "orig."), ("@para", "rew."), ("@paranq", "rew.-nq")]
    cols = ([(t, "P_L1_ap" + tag, "tpr_at_1fpr", "mean%") for tag, t in tags]
            + [(t, "A_L3_hard" + tag, "auroc", "mean") for tag, t in tags]
            + [(t, "A_L3_hard" + tag, "tpr_at_1fpr", "mean%") for tag, t in tags])
    hdr = (" & \\multicolumn{3}{c}{Poison L1, TPR@1\\%} & \\multicolumn{3}{c}{Activation L3, AUROC} & "
           "\\multicolumn{3}{c}{Activation L3, TPR@1\\%} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}")
    return tex_table(ms, cols, header=hdr)


def t_active():
    ms = ["cpmg_full", "cpmg_write_retrieval", "---", "abl_no_prov", "abl_no_query", "abl_no_nbrs", "write_only", "---",
          "content_only", "piguard", "keyword", "tfidf_memory", "tfidf_full", "---", "lexmatch", "lexquote"]
    cols = [("L1", "P_L1", "auroc"), ("L1 TPR", "P_L1", "tpr_at_1fpr", "%"), ("L2", "P_L2", "auroc"),
            ("no trig.", "A_L3_none_real", "auroc"), ("wrong", "A_L3_mis_real", "auroc"), ("overlap", "A_L3_ovl_real", "auroc"),
            ("TPR", "A_L3_hard_real", "tpr_at_1fpr", "%"),
            ("wrong", "A_L23_mis_real", "auroc"), ("overlap", "A_L23_ovl_real", "auroc"), ("TPR", "A_L23_hard_real", "tpr_at_1fpr", "%"),
            ("L3", "A_L3_hard_rand", "auroc")]
    hdr = (" & \\multicolumn{3}{c}{Poison} & \\multicolumn{4}{c}{Activation L3, real triggers} & "
           "\\multicolumn{3}{c}{Activation L2+L3, real} & Rand. \\\\\n"
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-8}\\cmidrule(lr){9-11}\\cmidrule(lr){12-12}")
    return tex_table(ms, cols, header=hdr)


def t_xtemplate():
    return tex_table(["cpmg_full", "cpmg_noehr", "content_only", "---", "lexquote", "lexmatch"],
                     [("P-L1", "P_L1_ehr", "auroc", "mean"), ("P-L2", "P_L2_ehr", "auroc", "mean"),
                      ("A-L2+L3", "A_L23_hard_ehr", "auroc", "mean"), ("A-L3", "A_L3_hard_ehr", "auroc", "mean")])


def t_data():
    srcs = ["AgentPoison", "ASB", "memshield-bench", "MemoryGraft"]
    L = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
         " & \\multicolumn{2}{c}{Train} & \\multicolumn{2}{c}{Val} & \\multicolumn{2}{c}{Test} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
         "Source & clean & pois. & clean & pois. & clean & pois. \\\\", "\\midrule"]
    for s in srcs:
        c = [f"{STATS[sp]['by_source_label'].get(f'{s}|{lab}', 0):,}" for sp in ("train", "val", "test") for lab in (0, 1)]
        L.append(f"{s} & " + " & ".join(c) + " \\\\")
    L.append("\\midrule")
    L.append("Total & " + " & ".join(f"{STATS[sp][k]:,}" for sp in ("train", "val", "test") for k in ("neg", "pos")) + " \\\\")
    return "\n".join(L + ["\\bottomrule", "\\end{tabular}"])


def t_pertrigger():
    names = [("ap", "\\texttt{com \\#\\#cturing reeve eased pulled}", "optimised (AgentPoison)"),
             ("ap_long", "\\texttt{kelsey rocked stench distinguish \\ldots}", "earlier optimised"),
             ("badchain", "\\texttt{In a parallel universe}", "BadChain phrase"),
             ("seed_ad", "\\texttt{Be careful and drive safe.}", "seed phrase"),
             ("seed_qa", "\\texttt{Make efficient tool calls.}", "seed phrase")]
    L = ["\\begin{tabular}{llrrr}", "\\toprule", "Trigger & Kind & $n$ & AUROC & TPR@1\\% \\\\", "\\midrule"]
    for k, t, kind in names:
        s = f"A_mis_{k}"
        L.append(f"{t} & {kind} & {SIZES[s]['n_pos']} & {cell('cpmg_full', s, 'auroc')} & {cell('cpmg_full', s, 'tpr_at_1fpr', '%')} \\\\")
    return "\n".join(L + ["\\bottomrule", "\\end{tabular}"])


def t_cost():
    rows = ["\\begin{tabular}{lrrr}", "\\toprule", "Model & Params & ms/check & Train (min) \\\\", "\\midrule"]
    for m in ["cpmg_full", "write_only", "content_only", "piguard", "tfidf_full", "lexmatch", "keyword"]:
        if m in RUNS:
            r = RUNS[m]
            p = r.get("n_params", 0)
            ps = f"{p/1e6:.0f}M" if p >= 1e6 else (f"{p/1e3:.0f}K" if p >= 1e3 else str(p))
            rows.append(f"{NAMES[m]} & {ps} & {r['latency_ms_bs1']:.2f} & "
                        + (f"{r['train_seconds']/60:.0f}" if r.get("train_seconds") else "--") + " \\\\")
    return "\n".join(rows + ["\\bottomrule", "\\end{tabular}"])


TABLES = {
    "robust": t_robust, "main": t_active, "xtemplate": t_xtemplate, "data": t_data, "pertrigger": t_pertrigger, "cost": t_cost,
    "xsource": lambda: tex_table(["cpmg_noasb", "tfidf_memory_noasb", "piguard", "keyword"],
                                 [("AUROC", "asb_xsource", "auroc"), ("TPR@1\\%", "asb_xsource", "tpr_at_1fpr", "%"),
                                  ("Recall", "asb_xsource", "recall", "%"), ("FPR", "asb_xsource", "fpr", "%")]),
    "falsealarm": lambda: tex_table(["cpmg_full", "cpmg_write_retrieval", "abl_no_prov", "content_only", "piguard",
                                     "tfidf_memory", "tfidf_full"],
                                    [("NotInject", "notinject", "fpr", "%"), ("Trigger, no payload", "P_trig_only", "fpr", "%")]),
    "provswap": lambda: tex_table(["cpmg_full", "abl_no_query", "abl_no_nbrs", "write_only"],
                                  [("Orig.", "P_all", "auroc", "mean"), ("Swapped", "P_all@provswap", "auroc", "mean"),
                                   ("Removed", "P_all@noprov", "auroc", "mean"),
                                   ("$|\\Delta|$", "provswap_delta", "mean_abs_delta", "mean"),
                                   ("Flips\\,\\%", "provswap_delta", "decision_flip_rate", "mean%")]),
    "persource": lambda: tex_table(["cpmg_full", "cpmg_write_retrieval", "write_only", "content_only", "piguard", "keyword",
                                    "tfidf_memory", "tfidf_full"],
                                   [(s, f"P_src_{k}", "auroc") for s, k in (("AgentPoison", "AgentPoison"), ("ASB", "ASB"),
                                                                            ("memshield", "memshield-bench"), ("MemoryGraft", "MemoryGraft"))]),
}


def resolve(mm):
    p = mm.group(1).split(":")
    try:
        if p[0] == "table":
            return TABLES[p[1]]()
        if p[0] == "stats":
            v = STATS
            for k in p[1:]:
                v = v[k]
            return f"{v:,}" if isinstance(v, int) else str(v)
        if p[0] == "run":
            v = RUNS[p[1]]
            for k in p[2].split("."):
                v = v[k]
            if len(p) > 3 and p[3] == "sci":
                m_, e_ = f"{v:.0e}".split("e")
                return f"${m_}\\times10^{{{int(e_)}}}$"
            return format(v, p[3]) if len(p) > 3 else str(v)
        if p[0] == "policy":
            return str(POLICY[p[1]]["totals"].get(p[2], 0))
        if p[0] == "para_example":
            return para_example(p[1])
        if p[0] == "para":
            return f"{PARA[p[1]][p[2]]:,}"
        if p[0] == "n":
            return f"{SIZES[p[1]][p[2]]:,}"
        if p[0] == "seeds":
            return str(max((a["n_seeds"] for s in AGG[p[1]].values() for a in s.values()), default=0))
        if p[0] == "ci":
            c = S42[p[1]][p[2]]["auroc_ci95"]
            return f"[{c[0]:.3f}, {c[1]:.3f}]"
        return val(p[0], p[1], p[2], p[3] if len(p) > 3 else "")
    except (KeyError, TypeError, IndexError) as e:
        raise SystemExit(f"UNRESOLVED placeholder [[{mm.group(1)}]]: {e!r}")


def compile_pdf(tex="paper.tex"):
    if shutil.which("tectonic"):
        return subprocess.run(["tectonic", tex], cwd=HERE).returncode
    cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex]
    if shutil.which("latexmk"):
        return subprocess.run(cmd, cwd=HERE, stdout=open(os.path.join(HERE, "build.log"), "w"), stderr=subprocess.STDOUT).returncode
    if shutil.which("docker"):
        d = ["docker", "run", "--rm", "--network", "none", "-u", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
             "-v", f"{HERE}:/work", "-w", "/work", "texlive/texlive:latest-full"] + cmd
        return subprocess.run(d, stdout=open(os.path.join(HERE, "build.log"), "w"), stderr=subprocess.STDOUT).returncode
    print("No LaTeX engine found: upload the paper/ folder to Overleaf and compile paper.tex.")
    return 1


def main():
    subprocess.run([os.path.join(ROOT, ".venv/bin/python"), os.path.join(HERE, "make_figures.py")], check=True)
    src = open(os.path.join(HERE, "paper.tex.in"), encoding="utf-8").read()
    out = re.sub(r"\[\[([^\[\]]+)\]\]", resolve, src)
    open(os.path.join(HERE, "paper.tex"), "w", encoding="utf-8").write(out)
    print("wrote paper/paper.tex")
    rc = compile_pdf()
    print("compile exit code:", rc, "(log: paper/build.log)")
    if "--review" in sys.argv:  # anonymous version for double-blind review: no authors, no repository link
        rev = out.replace("\\usepackage[final]{acl}", "\\usepackage[review]{acl}")
        a_, b_ = rev.index("\\author{"), rev.index("\\begin{document}")
        rev = rev[:a_] + "\\author{Anonymous submission}\n\n" + rev[b_:]
        rev = rev.replace("\\url{https://github.com/ShahmeerHyat/cpmg-memory-poisoning}",
                          "a public repository (link removed for anonymous review)")
        rev = rev.replace("submitted as a course deliverable on 8 November 2026, used", "used")
        assert "Shahmeer" not in rev and "23I-" not in rev and "FAST" not in rev and "Sohail" not in rev
        open(os.path.join(HERE, "paper_review.tex"), "w", encoding="utf-8").write(rev)
        print("review build exit code:", compile_pdf("paper_review.tex"))


if __name__ == "__main__":
    main()
