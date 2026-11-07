---
language: en
library_name: transformers
pipeline_tag: text-classification
base_model: microsoft/deberta-v3-base
tags: [security, memory-poisoning, llm-agents, prompt-injection, deberta-v3]
---

# CPMG — Contextual Provenance Memory Guard (course project checkpoints)

Two DeBERTa-v3-base sequence classifiers from a course project (Agentic AI, FAST-NUCES Islamabad;
Shahmeer Ali, Mohammad Soban, Ali Rauf). They score a stored LLM-agent memory for poisoning risk.
Code, data pipeline, full results and report: **https://github.com/ShahmeerHyat/cpmg-memory-poisoning**

| Folder | Model | Test AUROC (all) | L1 | L2 (derived) | L3 (derived) | L3, held-out trigger | ms/check (RTX 5060, bs 1) |
|---|---|---|---|---|---|---|---|
| `cpmg_full/` | CPMG (retrieval checkpoint) | 0.941 | 0.987 | 1.000 | 0.655 | 0.495 | 10.5 |
| `content_only/` | Content-only DeBERTa baseline | 0.815 | 0.999 | 0.500 | 0.498 | 0.499 | 10.4 |

All numbers come from `results/metrics.json` in the GitHub repo (single seed 42; real public sources, with
L2/L3 test cases *derived* from them). Only these two checkpoints are uploaded; the ablation, write-only and cross-source
models can be retrained with `./run_all_training.sh`.

## Input format
CPMG (`cpmg_full/`) expects
`[CLS] [CH_*][AUTH_*][AGE_*][PARENT_*] memory [SEP] source [SEP] query/action [SEP] nbr1 [SEP] nbr2 [SEP] nbr3 [SEP]`
built by `cpmg/features.py::encode(example, tokenizer, "full", 512)` in the GitHub repo (the provenance
values are special tokens already in the saved tokenizer). `content_only/` takes the memory text alone.
Output label 1 = poisoned; risk score = softmax probability of label 1.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
repo = "ShahmeerHyat/cpmg-memory-poisoning"
tok = AutoTokenizer.from_pretrained(repo, subfolder="content_only")
model = AutoModelForSequenceClassification.from_pretrained(repo, subfolder="content_only")
```

## Limitations (read before use)
- Research checkpoints, single seed, trained on a small mix of public sources (AgentPoison, ASB, MemoryGraft,
  memshield-bench). L2/L3 cases are derived from AgentPoison records.
- CPMG learned the specific AgentPoison trigger: on the held-out BadChain trigger its L3 AUROC is at chance.
- Trained without ASB, CPMG does not transfer to ASB; provenance tags were assigned per source.
- Not a production guard. See the report and `results/DECISIONS.md` in the GitHub repo.

Training data licenses: see `DATA_LICENSES.md` in the GitHub repo (includes CC-BY-SA-4.0 material).
