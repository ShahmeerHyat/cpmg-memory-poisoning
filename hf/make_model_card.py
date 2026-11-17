"""Write the Hugging Face model card (hf/README.md) with numbers read from results/metrics.json (dataset v2).

Usage: python hf/make_model_card.py            # writes hf/README.md
       python hf/make_model_card.py --upload   # also uploads the card and the two checkpoints to the Hub
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = json.load(open(os.path.join(ROOT, "results/metrics.json")))["over_seeds"]
RUNS = {m: json.load(open(os.path.join(ROOT, "results/runs", f"{m}.json"))) for m in ("cpmg_full", "content_only")}
GH = "https://github.com/ShahmeerHyat/cpmg-memory-poisoning"
REPO = "ShahmeerHyat/cpmg-memory-poisoning"


def f(m, s, k="auroc", pct=False):
    v = A[m][s][k]["mean"]
    return f"{100*v:.1f}" if pct else f"{v:.3f}"


rows = "\n".join(
    f"| `{sub}/` | {name} | {f(m,'P_L1')} | {f(m,'P_L2')} | {f(m,'A_L3_hard_real')} | {f(m,'A_L23_hard_real')} | "
    f"{f(m,'A_L3_hard@paranq')} | {f(m,'notinject','fpr',True)} | {RUNS[m]['latency_ms_bs1']:.1f} |"
    for sub, m, name in (("cpmg_full", "cpmg_full", "CPMG (retrieval checkpoint, seed 42)"),
                         ("content_only", "content_only", "Content-only DeBERTa baseline (seed 42)")))

card = f"""---
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

Code, data pipeline, full results and the paper (`paper/paper.pdf`): **{GH}**

| Folder | Model | Poison L1 | Poison L2 | Activation L3, real triggers | Activation L2+L3, real triggers | Activation L3, reworded attack | NotInject false alarms (%) | ms/check (RTX 5060, bs 1) |
|---|---|---|---|---|---|---|---|---|
{rows}

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
repo = "{REPO}"
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
"""
os.makedirs(os.path.join(ROOT, "hf"), exist_ok=True)
open(os.path.join(ROOT, "hf", "README.md"), "w").write(card)
print(card[:1400])

if "--upload" in sys.argv:
    from huggingface_hub import HfApi
    api = HfApi()
    for sub in ("cpmg_full", "content_only"):
        api.upload_folder(repo_id=REPO, folder_path=os.path.join(ROOT, "checkpoints", sub), path_in_repo=sub,
                          allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"],
                          commit_message=f"Dataset v2 checkpoint: {sub} (two outputs: poison, activation)")
        print("uploaded", sub)
    api.upload_file(repo_id=REPO, path_or_fileobj=os.path.join(ROOT, "hf", "README.md"), path_in_repo="README.md",
                    commit_message="Model card for dataset v2")
    print("uploaded model card")
