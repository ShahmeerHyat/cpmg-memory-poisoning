"""Input formatting for CPMG.

Full input (CLAUDE.md):
  [CLS] [provenance tokens] memory [SEP] source interaction [SEP] query/action [SEP] nbr1 [SEP] nbr2 [SEP] nbr3 [SEP]
Provenance values are added to the tokenizer as special tokens. Ablations drop whole segments.
Each segment is truncated to its own budget; the memory gets whatever budget is left.
"""

CHANNELS = ["email", "slack", "doc", "web", "calendar", "tool", "chat", "unknown"]
AUTHORITIES = ["user", "internal", "external", "unknown"]
AGES = ["unknown"]
PARENTS = ["kb_retrieval", "code_exec", "agent_plan", "setup_script", "none", "unknown"]

PROV_TOKENS = ([f"[CH_{c}]" for c in CHANNELS] + [f"[AUTH_{a}]" for a in AUTHORITIES]
               + [f"[AGE_{a}]" for a in AGES] + [f"[PARENT_{p}]" for p in PARENTS])

# Input variants (which segments each model sees)
VARIANTS = {
    "full":         dict(prov=True,  source=True,  query=True,  nbrs=True),
    "no_prov":      dict(prov=False, source=True,  query=True,  nbrs=True),
    "no_query":     dict(prov=True,  source=True,  query=False, nbrs=True),
    "no_nbrs":      dict(prov=True,  source=True,  query=True,  nbrs=False),
    "write_only":   dict(prov=True,  source=True,  query=False, nbrs=False),
    "content_only": dict(prov=False, source=False, query=False, nbrs=False),
}

BUDGET = dict(source=32, query=48, nbr=64, min_memory=128)


def prov_tokens(ex):
    return [f"[CH_{ex['channel']}]", f"[AUTH_{ex['authority']}]", f"[AGE_{ex['age']}]", f"[PARENT_{ex['parent']}]"]


def add_prov_tokens(tokenizer):
    return tokenizer.add_special_tokens({"additional_special_tokens": PROV_TOKENS})


def _ids(tok, text, n):
    if n <= 0 or not text:
        return []
    return tok(text, add_special_tokens=False, truncation=True, max_length=n)["input_ids"]


def encode(ex, tok, variant, max_len):
    v = VARIANTS[variant]
    cls, sep = tok.cls_token_id, tok.sep_token_id
    prov = tok.convert_tokens_to_ids(prov_tokens(ex)) if v["prov"] else []
    tail = []
    if v["source"]:
        tail += _ids(tok, ex["source"], BUDGET["source"]) + [sep]
    if v["query"]:
        tail += _ids(tok, ex["query"], BUDGET["query"]) + [sep]
    if v["nbrs"]:
        for n in ex["neighbours"]:
            tail += _ids(tok, n, BUDGET["nbr"]) + [sep]
    used = 1 + len(prov) + 1 + len(tail)  # cls + prov + mem-sep + tail
    mem_budget = max_len - used
    if mem_budget < BUDGET["min_memory"]:  # shrink tail rather than starve the memory
        tail = tail[: max(0, len(tail) - (BUDGET["min_memory"] - mem_budget))]
        mem_budget = max_len - (2 + len(prov) + len(tail))
    mem = _ids(tok, ex["memory"], mem_budget)
    ids = [cls] + prov + mem + [sep] + tail
    return ids[:max_len]


def as_text(ex, variant="full"):
    """Flat text version of the same input, for the TF-IDF baselines."""
    v = VARIANTS[variant]
    parts = []
    if v["prov"]:
        parts.append(" ".join(prov_tokens(ex)))
    parts.append(ex["memory"])
    if v["source"]:
        parts.append("SRC " + ex["source"])
    if v["query"]:
        parts.append("QRY " + ex["query"])
    if v["nbrs"]:
        parts += ["NBR " + n for n in ex["neighbours"]]
    return " || ".join(parts)
