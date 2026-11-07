"""Keyword/regex filter baseline (pointwise, memory text only, no training).

The pattern list is a generic prompt-injection cue list written before looking at test results;
it is not tuned to this dataset. Score = number of distinct patterns matched.
Usage: python -m baselines.keyword
"""
import re

from baselines.common import PRED_SPLITS, latency_ms, load, write_preds, write_run

PATTERNS = [
    r"ignore (all |the )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (all |the )?(previous|prior|above)",
    r"you must (strictly )?(adhere|follow|obey)",
    r"do not (use|tell|mention|reveal|inform)",
    r"(new|updated|override) (instructions?|policy|rules?)",
    r"system prompt",
    r"\b(jailbreak|bypass|override)\b",
    r"\b(password|api[_ ]?key|token|credential|secret)s?\b",
    r"\b(exfiltrat\w*|send (it|this|all|the data) to|forward (it|this|all) to)\b",
    r"https?://",
    r"\bcurl\b|\bwget\b",
    r"\b(delete|wipe|erase|drop) (all|the|every)\b",
    r"\b(disable|turn off) (security|logging|monitoring|validation|checks?)\b",
    r"\bskip (the )?(validation|review|checks?|verification)\b",
    r"answer: task complete",
    r"\bwithout (telling|informing|notifying|alerting)\b",
    r"\b(unnoticed|undetected|covertly|secretly)\b",
    r"\bact as\b|\bpretend (to be|you are)\b",
]
RX = [re.compile(p, re.I) for p in PATTERNS]


def score(rows):
    return [sum(1 for rx in RX if rx.search(r["memory"])) for r in rows]


def main():
    for sp in PRED_SPLITS:
        rows = load(sp)
        write_preds("keyword", sp, rows, score(rows))
    write_run("keyword", latency_ms_bs1=latency_ms(score, load("test")), n_params=0, n_patterns=len(PATTERNS),
              device="cpu", input="memory text only")
    print("keyword done")


if __name__ == "__main__":
    main()
