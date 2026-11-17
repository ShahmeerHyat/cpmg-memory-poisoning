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
N_BG = 1500

# Real triggers shipped in the AgentPoison repo @ 7b02e28f. TEST ONLY: never used in train or val (D22).
REAL_TRIGGERS = {
    "ap": "com ##cturing reeve eased pulled",                # ReAct/run_strategyqa_inference.py:143 (optimised)
    "ap_long": "kelsey rocked stench distinguish the useful blast realized problem rand ᅢ",  # :142 (earlier optimised)
    "badchain": "In a parallel universe",                    # :149 (BadChain)
    "seed_ad": "Be careful and drive safe.",                 # algo/trigger_optimization.py:621 (seed trigger)
    "seed_qa": "Make efficient tool calls.",                 # algo/trigger_optimization.py:623 (seed trigger)
}


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
    for k in keys[:N_BG]:
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


def bert_vocab():
    """BERT wordpiece vocabulary = the token space AgentPoison optimises its triggers over (DPR/BERT encoders)."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("google-bert/bert-base-uncased", "vocab.txt")
    real_toks = {t for v in REAL_TRIGGERS.values() for t in v.lower().split()}
    out = []
    for t in open(path, encoding="utf-8").read().split("\n"):
        core = t[2:] if t.startswith("##") else t
        if core.isascii() and core.isalpha() and len(core) >= 3 and t not in real_toks:
            out.append(t)
    return out


def assign_triggers(recs, seed):
    """D22. Re-template every AgentPoison pair with the trigger for its split.

    train/val: one unique random trigger per group. Half are random BERT-vocabulary token sequences (the form of
    AgentPoison's optimised triggers), half are 3-6 word spans cut from real StrategyQA paragraphs that are not used
    anywhere else in the dataset (the form of BadChain's phrase trigger). SYNTHETIC, training-side only.
    test: half of the groups get one of the five REAL shipped triggers, half get fresh random triggers.
    Each pair also gets a distractor trigger (another trigger of the same pool) for the mismatch negatives.
    """
    vocab = bert_vocab()
    paras = json.load(open(P("data/raw/AgentPoison/ReAct/database/strategyqa_train_paragraphs.json")))
    keys = sorted(paras)
    random.Random(seed).shuffle(keys)
    span_src = [paras[k]["content"].split() for k in keys[N_BG:]]  # paragraphs never used as records
    real_l = [v.lower() for v in REAL_TRIGGERS.values()]
    used = set()

    def random_trigger(rng):
        while True:
            if rng.random() < 0.5:
                kind, t = "random_vocab", " ".join(rng.choice(vocab) for _ in range(rng.randint(3, 8)))
            else:
                w = rng.choice(span_src)
                n = rng.randint(3, 6)
                if len(w) <= n:
                    continue
                i = rng.randint(0, len(w) - n)
                kind, t = "random_span", " ".join(w[i:i + n]).strip()
            tl = t.lower()
            if len(t) < 8 or t in used or any(x in tl or tl in x for x in real_l) or "'" in t or "’" in t:
                continue
            used.add(t)
            return kind, t

    names = sorted(REAL_TRIGGERS)
    pools = collections.defaultdict(list)
    for r in sorted((r for r in recs if r["kind"] == "ap_pair"), key=lambda r: r["p_id"]):
        rng = random.Random(f"{seed}:trig:{r['p_id']}")
        old = r["trigger"]
        if r["split"] == "test" and hfrac("real:" + r["group"], seed) < 0.5:
            name = names[int(hfrac("which:" + r["p_id"], seed) * len(names)) % len(names)]
            kind, new, fam, pool = "real", REAL_TRIGGERS[name], "ap_real_" + name, "test_real"
        else:
            kind, new = random_trigger(rng)
            fam, pool = "ap_" + kind, r["split"] + "_random"
        assert old in r["p_text"] and old in r["payload"] and old not in r["c_text"], r["p_id"]
        r["p_text"], r["payload"] = r["p_text"].replace(old, new), r["payload"].replace(old, new)
        r.update(trigger=new, trigger_kind=kind, family=fam, trig_pool=pool)
        pools[pool].append(r)
    for pool, rs in pools.items():
        for r in rs:
            rng = random.Random(f"{seed}:distr:{r['p_id']}")
            while True:
                d = rng.choice(rs)["trigger"]
                if d != r["trigger"]:
                    break
            r["distractor"] = d


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
            fam, q, trig, dis = r["family"], r["q"], r["trigger"], r["distractor"]
            gf = dict(common, subset=sub, group_family=fam, source=q, trigger=trig, trigger_kind=r["trigger_kind"])
            P_, C_ = r["p_text"], r["c_text"]
            n2 = list(n3[:N_NBRS - 1])  # payload stored as a separate co-retrieved record
            n2.insert(rng.randint(0, len(n2)), r["payload"])
            qx, qy = q + " " + trig, q + " " + dis  # authors' format: trigger appended to the question
            # benign overlap (D29): the query repeats a 3-6 word span of the record's own clean text
            srng = random.Random(f"{seed}:span:{r['p_id']}")
            w = C_.split()
            lo = min(len(("Question: " + q).split()) + 1, max(0, len(w) - 8))
            span = None
            for _ in range(200):
                n = srng.randint(3, 6)
                i = srng.randint(lo, max(lo, min(len(w), lo + 150) - n))
                cand = " ".join(w[i:i + n])
                if len(cand) >= 8 and cand in P_ and cand in C_ and cand.lower() not in trig.lower() \
                        and trig.lower() not in cand.lower():
                    span = cand
                    break
            assert span, r["p_id"]
            qo = q + " " + span

            def add(variant, level, yp, ya, memory, query, nbrs, sid, derived, qt):
                exs.append(dict(gf, variant=variant, level=level, label=yp, y_poison=yp, y_active=ya,
                                attack_family=fam if yp else "none", memory=memory, query=query, neighbours=nbrs,
                                source_id=sid, derived=derived, query_trigger=qt))
            # y_poison: attacker-written content is in the context (memory or neighbours).
            # y_active: the query carries the trigger that the poisoned content in the context is conditional on.
            add("L1_pos", "L1", 1, 0, P_, q, n3, r["p_id"], False, "none")       # real poisoned record, neutral query
            add("L1_twin", "L1", 0, 0, C_, q, n3, r["c_id"], False, "none")      # its clean twin
            add("L2_pos", "L2", 1, 0, C_, q, n2, r["p_id"], True, "none")        # payload only among the neighbours
            add("L3_pos", "L3", 1, 1, P_, qx, n3, r["p_id"], True, "match")      # dormant record, its trigger arrives
            add("L3_mis", "L3", 1, 0, P_, qy, n3, r["p_id"], True, "mismatch")   # a different trigger arrives
            add("L23_pos", "L23", 1, 1, C_, qx, n2, r["p_id"], True, "match")    # payload in neighbour + its trigger
            add("L23_mis", "L23", 1, 0, C_, qy, n2, r["p_id"], True, "mismatch")
            add("L3_ovl", "L3", 1, 0, P_, qo, n3, r["p_id"], True, "benign_overlap")   # repeats harmless record text
            add("L23_ovl", "L23", 1, 0, C_, qo, n2, r["p_id"], True, "benign_overlap")
            add("trig_only", "bgq", 0, 0, C_, qx, n3, r["c_id"], True, "no_payload")  # trigger, but nothing to activate
        else:
            sd = r["source_dataset"]
            sub = "strategyqa" if r["subset"] == "strategyqa_paragraph" else ""
            sp = common["split"]
            n3 = nb(sd, sub, sp, r["text"], r["source_id"])
            ya = 0 if r["subset"] == "strategyqa_paragraph" else -1  # activation is only defined for AgentPoison
            exs.append(dict(common, subset=r["subset"], group_family=r["family"], source=r["src"], level=r["level"],
                            label=r["label"], y_poison=r["label"], y_active=ya, trigger=None, trigger_kind=None,
                            query_trigger="none", attack_family=r["family"], memory=r["text"], query=r["q"],
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
                            label=0, y_poison=0, y_active=-1, trigger=None, trigger_kind=None, query_trigger="none", attack_family="none", memory=row["prompt"], query="[none]", neighbours=[],
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
    assign_triggers(recs, a.seed)
    exs = build_examples(recs, a.seed)
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
            by_variant={k: int(v) for k, v in df.variant.value_counts().items()},
            by_trigger_kind={str(k): int(v) for k, v in df[df.variant == "L3_pos"].trigger_kind.value_counts().items()},
            by_real_trigger={k: int(v) for k, v in df[(df.variant == "L3_pos") & (df.trigger_kind == "real")]
                             .group_family.value_counts().items()},
            n_unique_triggers=int(df.trigger.dropna().nunique()),
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
    def has_real(e):
        t = " ".join([e["memory"], e["query"]] + e["neighbours"]).lower()
        return any(v.lower() in t for v in REAL_TRIGGERS.values())
    stats["real_triggers_in_train_or_val"] = sum(1 for n in ("train", "val") for e in splits[n] if has_real(e))
    trig = {n: {e["trigger"] for e in splits[n] if e.get("trigger")} for n in ("train", "val", "test")}
    stats["triggers_shared_train_test"] = len(trig["train"] & trig["test"])
    stats["triggers_shared_train_val"] = len(trig["train"] & trig["val"])
    stats["triggers_shared_val_test"] = len(trig["val"] & trig["test"])
    # no input may appear twice with different labels (the v1 L1_pos / L3_twin conflict)
    for n in ("train", "val", "test"):
        seen = collections.defaultdict(set)
        for e in splits[n]:
            seen[(e["memory"], e["query"], tuple(e["neighbours"]), e["source"])].add((e["y_poison"], e["y_active"]))
        stats[f"{n}_inputs_with_conflicting_labels"] = sum(1 for v in seen.values() if len(v) > 1)
        stats[f"{n}_duplicate_inputs"] = len(splits[n]) - len(seen)
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: v[kk] for kk in ("n", "pos", "neg", "groups")})
                      for k, v in stats.items()}, indent=1))

    # 5 examples per level for the sanity check
    lines = ["# Data examples v2 (5 per variant)\n"]
    rng = random.Random(a.seed)
    for sp_name, var in [("train", v) for v in ("L1_pos", "L1_twin", "L2_pos", "L3_pos", "L3_mis", "L23_pos",
                                                 "L23_mis", "L3_ovl", "L23_ovl", "trig_only", "L1_neg", "bg")] + [("test", "L3_pos"), ("test", "L3_mis")]:
        for lab in (None,):
            pool = [e for e in splits[sp_name] if e["variant"] == var]
            if not pool:
                continue
            lines.append(f"\n## {var} ({sp_name}; {len(pool)} examples; y_poison={pool[0]['y_poison']}, "
                         f"y_active={pool[0]['y_active']})\n")
            for e in rng.sample(pool, min(5, len(pool))):
                lines.append(f"\n**{e['id']}** source={e['source_dataset']}/{e['subset']} family={e['attack_family']} "
                             f"prov={e['channel']}/{e['authority']}/{e['parent']} derived={e['derived']} trigger={e.get('trigger')!r}\n")
                lines.append(f"- memory: `{e['memory'][-420:]!r}`\n- query: `{e['query'][:260]!r}`\n")
                for n in e["neighbours"]:
                    lines.append(f"- neighbour: `{n[:160]!r}`\n")
    os.makedirs(P("results"), exist_ok=True)
    open(P("results/data_examples.md"), "w").write("".join(lines))


if __name__ == "__main__":
    main()
