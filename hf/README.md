---
language: en
library_name: transformers
pipeline_tag: text-classification
base_model: microsoft/deberta-v3-base
tags: [security, memory-poisoning, llm-agents, prompt-injection, deberta-v3]
---

# CPMG: Contextual Provenance Memory Guard (research checkpoints, dataset v2)

Two DeBERTa-v3-base cross-encoders from a course project (Agentic AI, FAST-NUCES Islamabad;
Shahmeer Ali, Mohammad Soban, Ali Rauf). They score a stored LLM-agent memory record with **two outputs**:

- `poison`: attacker-written content is in the context (the record or a co-retrieved neighbour);
- `activation`: the current query carries the trigger that content is conditional on.

Code, data pipeline, full results and the paper (`paper/paper.pdf`): **https://github.com/ShahmeerHyat/cpmg-memory-poisoning**

| Folder | Model | Poison L1 | Poison L2 | Activation L3, real triggers | Activation L2+L3, real triggers | Activation L3, reworded attack | NotInject false alarms (%) | ms/check (RTX 5060, bs 1) |
|---|---|---|---|---|---|---|---|---|
| `cpmg_full/` | CPMG (retrieval checkpoint, seed 42) | 1.000 | 1.000 | 1.000 | 1.000 | 0.998 | 22.2 | 10.5 |
| `content_only/` | Content-only DeBERTa baseline (seed 42) | 0.993 | 0.499 | 0.504 | 0.499 | 0.500 | 43.0 | 10.4 |

Test AUROC unless marked; values are means over three training seeds, the uploaded weights are seed 42. Activation is
measured against wrong-trigger and benign-overlap negatives; the real triggers are the five shipped with AgentPoison and
were never seen in training. All numbers come from `results/metrics.json` in the GitHub repo. L2/L3 cases are *derived* from
AgentPoison records. Only these two checkpoints are uploaded; the others can be retrained with `./start_training.sh`.

These weights replace the earlier (8 Nov 2026, dataset v1) upload, which had a single risk output; the v1 files remain in
this repository's commit history.

## Input format
CPMG (`cpmg_full/`) expects
`[CLS] [CH_*][AUTH_*][AGE_*][PARENT_*] memory [SEP] source [SEP] query/action [SEP] nbr1 [SEP] nbr2 [SEP] nbr3 [SEP]`
built by `cpmg/features.py::encode(example, tokenizer, "full", 512)` in the GitHub repo (the provenance values are special
tokens already in the saved tokenizer). `content_only/` takes the memory text alone
(`encode(example, tokenizer, "content_only", 512)`).

The head has two logits. Apply a **sigmoid** to each (not a softmax): index 0 = poison, index 1 = activation.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
repo = "ShahmeerHyat/cpmg-memory-poisoning"
tok = AutoTokenizer.from_pretrained(repo, subfolder="cpmg_full")
model = AutoModelForSequenceClassification.from_pretrained(repo, subfolder="cpmg_full").eval()
# ids = encode(example, tok, "full", 512) from cpmg/features.py
# poison, activation = torch.sigmoid(model(input_ids=torch.tensor([ids])).logits)[0].tolist()
```

## Limitations (read before use)
- Research checkpoints trained on a small mix of public sources (AgentPoison, ASB, MemoryGraft, memshield-bench).
  All compositional and dormant cases derive from AgentPoison's two injection templates; training triggers are synthetic.
- The model ignores its provenance tokens, and it mislabels activation on a record layout it has not seen
  (see the paper, Sections 7.3 and 7.4).
- False-alarm rate on out-of-distribution benign prompts (NotInject) is high; thresholds do not transfer across sources.
- Not a production guard. See the paper's Limitations section and `results/DECISIONS.md` in the GitHub repo.

Training data licenses: see `DATA_LICENSES.md` in the GitHub repo (includes CC-BY-SA-4.0 material).
