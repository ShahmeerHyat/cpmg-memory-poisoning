"""Build the CPMG dataset (L1 real, L2/L3 derived) and grouped splits from real public sources.

Inputs (produced in step 2):
  data/interim/agentpoison.jsonl, data/interim/asb_poisoned.jsonl, data/interim/asb_clean.jsonl
  data/raw/memshield-bench, data/raw/MemoryGraft, data/raw/NotInject, data/raw/AgentPoison (paragraphs)
Outputs:
  data/processed/{train,val,test,test_asb_xsource,test_notinject}.jsonl, data/processed/stats.json,
  results/data_examples.md
All decisions are documented in results/DECISIONS.md (D4-D13).

Usage: python data/build_dataset.py --seed 42
"""
import argparse
import collections
import hashlib
import json
import os
import random

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)

PROV = {  # D4
    "agentpoison_sq": ("web", "external", "kb_retrieval"),
    "agentpoison_ehr": ("tool", "internal", "code_exec"),
    "asb": ("tool", "user", "agent_plan"),
    "memorygraft": ("doc", "external", "setup_script"),
    "memshield": ("unknown", "unknown", "unknown"),
    "notinject": ("chat", "user", "none"),
}
N_NBRS = 3
HELDOUT_FAMILY = "badchain"


def jl(path):
    return [json.loads(l) for l in open(path)]


def hfrac(key, seed):
    return int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def split_of(group, seed):
    h = hfrac(group, seed)
    return "train" if h < 0.70 else ("val" if h < 0.85 else "test")


def base(source_key, **kw):
    ch, au, pa = PROV[source_key]
    r = dict(channel=ch, authority=au, parent=pa, age="unknown", derived=False, neighbours=[], notes="")
    r.update(kw)
    return r


def load_records(seed):
    """Return list of base records (one per real record) with group, label, texts."""
    recs = []
    # --- AgentPoison (StrategyQA + EHR): poisoned P and clean twin C ---
    ap = jl(P("data/interim/agentpoison.jsonl"))
    by_id = {r["source_id"]: r for r in ap}
    for r in ap:
        if r["label"] != 1:
            continue
        c = by_id[r["twin_id"]]
        sk = "agentpoison_sq" if r["subset"] == "strategyqa" else "agentpoison_ehr"
        q = r["orig_question"].strip()
        group = ("sq::" + r["record_key"].split("::")[1]) if sk == "agentpoison_sq" else ("ehr::" + q.lower())
        if sk == "agentpoison_sq":
            assert r["content"].startswith(c["content"]), r["source_id"]
            payload = r["content"][len(c["content"]):].strip()
        else:
            know_line = r["content"].split("\nSolution:")[0].split("\n")[-1]
            line = know_line[know_line.index("- Because there is"):]
            code = r["content"].split("\nSolution:", 1)[1]
            payload = f"Knowledge:\n{line}\nSolution:{code}"
        recs.append(base(sk, kind="ap_pair", group=group, source_dataset="AgentPoison", subset=r["subset"],
                         domain=r["domain"], family=r["attack_family"], trigger=r["trigger"], q=q,
                         p_text=r["content"], c_text=c["content"], payload=payload,
                         p_id=r["source_id"], c_id=c["source_id"]))
    # --- StrategyQA paragraphs (clean background, same KB as the ReAct agent) ---
    paras = json.load(open(P("data/raw/AgentPoison/ReAct/database/strategyqa_train_paragraphs.json")))
    keys = sorted(paras)
    random.Random(seed).shuffle(keys)
    for k in keys[:1500]:
        v = paras[k]
        recs.append(base("agentpoison_sq", kind="single", group="para::" + v["title"], source_dataset="AgentPoison",
                         subset="strategyqa_paragraph", domain="general_knowledge", family="none", label=0,
                         level="bg", text=v["content"].strip(), q=v["title"], src=v["title"],
                         source_id=f"agentpoison::paragraph::{k}"))
    # --- ASB poisoned (L1) and clean runs ---
    for r in jl(P("data/interim/asb_poisoned.jsonl")):
        recs.append(base("asb", kind="single", group="asb::" + r["agent"], source_dataset="ASB", subset="dpi_memory",
                         domain=r["agent"], family=r["attack_family"], label=1, level="L1", text=r["content"],
                         q=r["orig_question"], src=r["orig_question"], source_id=r["source_id"], agent=r["agent"],
                         attacker_instruction=r["attacker_instruction"], attacker_tool=r["attacker_tool"]))
    seen = set()
    for r in jl(P("data/interim/asb_clean.jsonl")):
        if r["content"] in seen:  # identical repeated samples
            continue
        seen.add(r["content"])
        recs.append(base("asb", kind="single", group="asb::" + r["agent"], source_dataset="ASB", subset="clean_run",
                         domain=r["agent"], family="none", label=0, level="L1", text=r["content"],
                         q=r["orig_question"], src=r["orig_question"], source_id=r["source_id"], agent=r["agent"]))
    # --- memshield-bench (minus AgentPoison copies, D5) ---
    for r in jl(P("data/raw/memshield-bench/data/full.jsonl")):
        if r["source"].startswith("agentpoison"):
            continue
        g = "ms::" + min(r["id"], r["paired_id"]) if r.get("paired_id") else "ms::" + r["id"]
        lab = int(r["label"] == "poisoned")
        recs.append(base("memshield", kind="single", group=g, source_dataset="memshield-bench", subset=r["source"],
                         domain=r["domain"], family=("memshield_" + r["attack_type"]) if lab else "none", label=lab,
                         level="L1", text=r["content"].strip(), q="[none]", src="[none]",
                         source_id="memshield::" + r["id"]))
    # --- MemoryGraft ---
    mg = json.load(open(P("data/raw/MemoryGraft/memorygraft/payloads/experience_seeds.json")))
    for key, lab in (("benign_experiences", 0), ("poisoned_experiences", 1)):
        for r in mg[key]:
            recs.append(base("memorygraft", kind="single", group="mg::" + r["id"], source_dataset="MemoryGraft",
                             subset=key, domain="data_science", family="memorygraft" if lab else "none", label=lab,
                             level="L1", text=f"{r['req']}\n{r['resp']}\ntags: {r['tag']}", q=r["req"], src=r["req"],
                             source_id="memorygraft::" + r["id"]))
    return recs


def assign_splits(recs, seed):
    # ASB has only 10 groups (agents): fixed 6/2/2 by seeded hash order so every split has some.
    asb_groups = sorted({r["group"] for r in recs if r["source_dataset"] == "ASB"}, key=lambda g: hfrac(g, seed))
    asb_split = {g: ("train" if i < 6 else "val" if i < 8 else "test") for i, g in enumerate(asb_groups)}
    for r in recs:
        r["split"] = asb_split.get(r["group"]) or split_of(r["group"], seed)


class NbrIndex:
    """TF-IDF retrieval over clean records of one (source key, split) pool."""

    def __init__(self, items):
        self.items = items  # list of (pair_key, text)
        self.vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 1), sublinear_tf=True)
        self.M = self.vec.fit_transform([t for _, t in items]) if items else None

    def get(self, text, exclude_key, k):
        if not self.items:
            return []
        sims = (self.M @ self.vec.transform([text]).T).toarray().ravel()
        out = []
        for i in np.argsort(-sims):
            key, t = self.items[i]
            if key == exclude_key or t == text:
                continue
            out.append(t)
            if len(out) == k:
                break
        return out


def pool_key(r):
    return (r["source_dataset"], r["subset"].split("_")[0] if r["source_dataset"] == "AgentPoison" else "", r["split"])


def build_examples(recs, seed, asb_all_split=None):
    rng = random.Random(seed)
    pools = collections.defaultdict(list)
    for r in recs:
        sd, sub = r["source_dataset"], r["subset"]
        sp = asb_all_split or r["split"]
        if r["kind"] == "ap_pair":
            pools[("AgentPoison", sub, r["split"])].append((r["group"], r["c_text"]))
        elif r["label"] == 0:
            key_sub = "strategyqa" if sub == "strategyqa_paragraph" else ""
            sp = asb_all_split if (asb_all_split and sd == "ASB") else r["split"]
            pools[(sd, key_sub, sp)].append((r.get("source_id"), r["text"]))
    index = {k: NbrIndex(v) for k, v in pools.items()}

    def nb(sd, sub, sp, text, excl, k=N_NBRS):
        return index[(sd, sub, sp)].get(text, excl, k) if (sd, sub, sp) in index else []

    exs = []
    for r in recs:
        common = {k: r[k] for k in ("group", "split", "source_dataset", "channel", "authority", "parent", "age", "domain")}
        if asb_all_split and r["source_dataset"] == "ASB":
            common["split"] = asb_all_split
        if r["kind"] == "ap_pair":
            sub, sp = r["subset"], r["split"]
            n3 = nb("AgentPoison", sub, sp, r["c_text"], r["group"])
            fam, q, trig = r["family"], r["q"], r["trigger"]
            gf = dict(common, subset=sub, group_family=fam, source=q)
            # L1: real poisoned record vs its clean twin; neutral query (D8)
            exs.append(dict(gf, level="L1", label=1, attack_family=fam, memory=r["p_text"], query=q, neighbours=n3,
                            source_id=r["p_id"], derived=False, variant="L1_pos"))
            exs.append(dict(gf, level="L1", label=0, attack_family="none", memory=r["c_text"], query=q, neighbours=n3,
                            source_id=r["c_id"], derived=False, variant="L1_twin"))
            # L2 (derived, D7): clean-looking memory; payload stored as a separate co-retrieved record
            n2 = list(n3[:N_NBRS - 1])
            n2.insert(rng.randint(0, len(n2)), r["payload"])
            exs.append(dict(gf, level="L2", label=1, attack_family=fam, memory=r["c_text"], query=q, neighbours=n2,
                            source_id=r["p_id"], derived=True, variant="L2_pos"))
            exs.append(dict(gf, level="L2", label=0, attack_family="none", memory=r["c_text"], query=q, neighbours=n3,
                            source_id=r["c_id"], derived=True, variant="L2_twin"))
            # L3 (derived, D7): same poisoned memory; trigger only in the later query
            exs.append(dict(gf, level="L3", label=1, attack_family=fam, memory=r["p_text"], query=q + " " + trig,
                            neighbours=n3, source_id=r["p_id"], derived=True, variant="L3_pos"))
            exs.append(dict(gf, level="L3", label=0, attack_family="none", memory=r["p_text"], query=q,
                            neighbours=n3, source_id=r["p_id"], derived=True, variant="L3_twin"))
        else:
            sd = r["source_dataset"]
            sub = "strategyqa" if r["subset"] == "strategyqa_paragraph" else ""
            sp = common["split"]
            n3 = nb(sd, sub, sp, r["text"], r["source_id"])
            exs.append(dict(common, subset=r["subset"], group_family=r["family"], source=r["src"], level=r["level"],
                            label=r["label"], attack_family=r["family"], memory=r["text"], query=r["q"],
                            neighbours=n3, source_id=r["source_id"], derived=False,
                            variant=("L1_pos" if r["label"] else ("bg" if r["level"] == "bg" else "L1_neg"))))
    for i, e in enumerate(exs):
        e["id"] = f"ex{i:06d}"
        e["scenario_id"] = e["group"]
    return exs


def notinject_examples():
    out = []
    for n in ("NotInject_one", "NotInject_two", "NotInject_three"):
        d = pd.read_parquet(P(f"data/raw/NotInject/data/{n}-00000-of-00001.parquet"))
        for i, row in d.iterrows():
            ch, au, pa = PROV["notinject"]
            out.append(dict(id=f"ni_{n}_{i}", group=f"ni::{n}::{i}", scenario_id=f"ni::{n}::{i}", split="test_notinject",
                            source_dataset="NotInject", subset=n, channel=ch, authority=au, parent=pa, age="unknown",
                            domain=row["category"], group_family="none", source="[none]", level="overdefense",
                            label=0, attack_family="none", memory=row["prompt"], query="[none]", neighbours=[],
                            source_id=f"notinject::{n}::{i}", derived=False, variant="overdefense"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=P("data/processed"))
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    recs = load_records(a.seed)
    assign_splits(recs, a.seed)
    exs = build_examples(recs, a.seed)

    # D11: held-out family removed from train/val entirely (whole groups)
    exs = [e for e in exs if not (e["split"] in ("train", "val") and e["group_family"] == HELDOUT_FAMILY)]
    # D12: ASB poisoned downsampled per agent to the clean count, in train and val
    rng = random.Random(a.seed)
    keep = []
    for sp in ("train", "val"):
        for g in sorted({e["group"] for e in exs if e["source_dataset"] == "ASB" and e["split"] == sp}):
            pos = [e for e in exs if e["group"] == g and e["label"] == 1]
            n_neg = sum(1 for e in exs if e["group"] == g and e["label"] == 0)
            keep += rng.sample(pos, min(n_neg, len(pos)))
    keep_ids = {e["id"] for e in keep}
    exs = [e for e in exs if not (e["source_dataset"] == "ASB" and e["split"] in ("train", "val")
                                   and e["label"] == 1 and e["id"] not in keep_ids)]

    # D13: cross-source ASB test set (all ASB records, neighbours from all ASB clean records)
    asb_recs = [r for r in recs if r["source_dataset"] == "ASB"]
    xs = build_examples(asb_recs, a.seed, asb_all_split="test_asb_xsource")
    for e in xs:
        e["id"] = "xs_" + e["id"]

    os.makedirs(a.out, exist_ok=True)
    splits = collections.defaultdict(list)
    for e in exs:
        splits[e["split"]].append(e)
    splits["test_asb_xsource"] = xs
    splits["test_notinject"] = notinject_examples()
    stats = {}
    for name, rows in splits.items():
        random.Random(a.seed).shuffle(rows)
        with open(os.path.join(a.out, f"{name}.jsonl"), "w") as f:
            for e in rows:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        df = pd.DataFrame(rows)
        stats[name] = dict(
            n=len(rows), pos=int(df.label.sum()), neg=int((1 - df.label).sum()), groups=int(df.group.nunique()),
            by_level_label={f"{k[0]}|{k[1]}": int(v) for k, v in df.groupby(["level", "label"]).size().items()},
            by_source_label={f"{k[0]}|{k[1]}": int(v) for k, v in df.groupby(["source_dataset", "label"]).size().items()},
            by_family={k: int(v) for k, v in df[df.label == 1].attack_family.value_counts().items()},
            by_provenance_label={f"{k[0]}/{k[1]}|{k[2]}": int(v) for k, v in
                                 df.groupby(["channel", "authority", "label"]).size().items()},
        )
    # leakage checks
    gsplit = collections.defaultdict(set)
    for name in ("train", "val", "test"):
        for e in splits[name]:
            gsplit[e["group"]].add(name)
    stats["groups_crossing_splits"] = sum(1 for v in gsplit.values() if len(v) > 1)
    tr_mem = {e["memory"] for e in splits["train"]}
    stats["test_memories_seen_in_train"] = sum(1 for e in splits["test"] if e["memory"] in tr_mem)
    stats["heldout_family_in_train_or_val"] = sum(1 for n in ("train", "val") for e in splits[n]
                                                 if e["group_family"] == HELDOUT_FAMILY)
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: v[kk] for kk in ("n", "pos", "neg", "groups")})
                      for k, v in stats.items()}, indent=1))

    # 5 examples per level for the sanity check
    lines = ["# Data examples (5 per level, from train)\n"]
    rng = random.Random(a.seed)
    for lvl in ("L1", "L2", "L3", "bg"):
        for lab in (1, 0):
            pool = [e for e in splits["train"] if e["level"] == lvl and e["label"] == lab]
            if not pool:
                continue
            lines.append(f"\n## {lvl}, label={lab} ({len(pool)} in train)\n")
            for e in rng.sample(pool, min(5, len(pool))):
                lines.append(f"\n**{e['id']}** source={e['source_dataset']}/{e['subset']} family={e['attack_family']} "
                             f"prov={e['channel']}/{e['authority']}/{e['parent']} derived={e['derived']}\n")
                lines.append(f"- memory: `{e['memory'][:400]!r}`\n- query: `{e['query'][:200]!r}`\n")
                for n in e["neighbours"]:
                    lines.append(f"- neighbour: `{n[:160]!r}`\n")
    os.makedirs(P("results"), exist_ok=True)
    open(P("results/data_examples.md"), "w").write("".join(lines))


if __name__ == "__main__":
    main()
