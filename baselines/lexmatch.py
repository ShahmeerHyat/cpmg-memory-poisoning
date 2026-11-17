"""Lexical copy rule for the activation target (no training, no model).

Idea: a dormant record names its trigger, so an activating query repeats a string that is in the context.
Score = share of the "extra" query tokens (query tokens beyond the source question) covered by the longest
contiguous run that also occurs in the memory or its neighbours. It knows nothing about the injection template.
This is the honest lower bar for the trigger-matching test: if this rule solves it, a learned model that
solves it has learned no more than string copying. The same score is written for the poison target, where it
is not expected to work.
Usage: python -m baselines.lexmatch
"""
import re

from baselines.common import PRED_SPLITS, latency_ms, load, write_preds, write_run

TOK = re.compile(r"[\w#]+", re.UNICODE)


def toks(t):
    return TOK.findall((t or "").lower())


def score_one(r):
    q, src = toks(r["query"]), toks(r["source"])
    extra = q[len(src):] if q[:len(src)] == src else q
    if not extra:
        return 0.0
    ctx = " " + " ".join(toks(r["memory"] + " \n " + " \n ".join(r["neighbours"]))) + " "
    best = 0
    for i in range(len(extra)):
        j = i + best + 1
        while j <= len(extra) and (" " + " ".join(extra[i:j]) + " ") in ctx:
            best = j - i
            j += 1
    return best / len(extra)


QUOTED = re.compile(r"[’'‘]([^’'‘\n]{3,120})[’'‘] in the question")


def score_one_quoted(r):
    """Same rule, but only strings the context puts in quotes before "in the question" count.
    This uses knowledge of the two AgentPoison injection templates, so it is an oracle-style reference, not a
    deployable defence: it cannot fire on an attack that words its condition differently."""
    q, src = toks(r["query"]), toks(r["source"])
    extra = q[len(src):] if q[:len(src)] == src else q
    if not extra:
        return 0.0
    e = " " + " ".join(extra) + " "
    best = 0
    for m in QUOTED.findall(r["memory"] + " \n " + " \n ".join(r["neighbours"])):
        t = toks(m)
        if t and (" " + " ".join(t) + " ") in e:
            best = max(best, len(t))
    return best / len(extra)


def score(rows):
    return [score_one(r) for r in rows]


def score_quoted(rows):
    return [score_one_quoted(r) for r in rows]


def main():
    for name, fn in (("lexmatch", score), ("lexquote", score_quoted)):
        for sp in PRED_SPLITS:
            rows = load(sp)
            write_preds(name, sp, rows, fn(rows))
        write_run(name, latency_ms_bs1=latency_ms(fn, load("test")), n_params=0, device="cpu",
                  input="query, source, memory, neighbours")
        print(name, "done")


if __name__ == "__main__":
    main()
