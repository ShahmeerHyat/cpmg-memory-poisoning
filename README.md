# CPMG — Contextual Provenance Memory Guard

Course project, Agentic AI (AI-4015), FAST-NUCES Islamabad.
Group: Shahmeer Ali (23I-0119), Mohammad Soban (23I-0056), Ali Rauf (23I-0076).

- Code and results: https://github.com/ShahmeerHyat/cpmg-memory-poisoning
- Model checkpoints: https://huggingface.co/ShahmeerHyat/cpmg-memory-poisoning
- Report: `report/report.pdf` · Data licenses: `DATA_LICENSES.md`

CPMG is a DeBERTa-v3-base cross-encoder that scores a stored agent memory for poisoning risk using
the memory, its source interaction, the current query/action, up to 3 co-retrieved memories and
system-recorded provenance tokens, plus a deterministic authority policy that stops low-authority
memory from authorising protected actions. Everything here uses **real public data only**
(see `data/raw/SOURCES.md`); L2/L3 cases are **derived** by transforming real records.

## Layout
```
data/sources/      scripts that produce records from the raw sources (AgentPoison templates, ASB export, ASB clean run)
data/build_dataset.py   unified dataset, L2/L3 derivation, grouped splits -> data/processed/*.jsonl
cpmg/              features.py (input formatting, provenance tokens), model.py, policy.py (authority policy)
baselines/         content_only.py, piguard.py, keyword.py, tfidf.py (TF-IDF+LR and provenance-only leak check)
train.py eval.py   training (all DeBERTa variants) and evaluation (all models, tables, plots)
configs/*.yaml     one config per DeBERTa run
results/           metrics.csv/json, tables.md, plots/, preds/, runs/, logs/, DECISIONS.md, PROGRESS.md
report/            report source (Markdown + generated LaTeX) and report.pdf
hf/                Hugging Face model card generator
```

## Reproduce

Hardware used: RTX 5060 Laptop (8 GB, Blackwell sm_120), 15 GB RAM, Ubuntu. Seed 42 everywhere.

```bash
# 0. Environment (torch 2.14.1 + CUDA 13.0 wheels; any Blackwell-capable build works)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python -r requirements.txt

# 1. Raw sources (shallow clones + HF datasets; commits pinned in data/raw/SOURCES.md)
mkdir -p data/raw && cd data/raw
git clone --depth 1 https://github.com/AI-secure/AgentPoison.git   # commit 7b02e28f
git clone --depth 1 https://github.com/agiresearch/ASB.git          # commit 544540ff
git clone --depth 1 https://github.com/Jacobhhy/MemoryGraft.git     # commit 82366205
cd ../..
.venv/bin/python -c "from huggingface_hub import snapshot_download as s; \
  s('npow/memshield-bench', repo_type='dataset', local_dir='data/raw/memshield-bench'); \
  s('leolee99/NotInject', repo_type='dataset', local_dir='data/raw/NotInject')"

# 2. Records from the sources
.venv/bin/python -I data/sources/agentpoison_inject.py      # AgentPoison templates -> data/interim/agentpoison.jsonl
.venv/bin/python -I data/sources/asb_export_poisoned.py     # ASB DPI memory DBs  -> data/interim/asb_poisoned.jsonl
#    ASB clean runs need a local LLM server (we used Qwen3-8B-Q4_K_M with llama.cpp):
#    llama-server -m Qwen3-8B-Q4_K_M.gguf --host 127.0.0.1 --port 8091 -ngl 99 -c 8192 --jinja
.venv/bin/python -I data/sources/asb_clean_run.py --samples 5   # -> data/interim/asb_clean.jsonl
#    (the produced data/interim/*.jsonl are committed, so this step can be skipped)

# 3. Dataset and splits
.venv/bin/python data/build_dataset.py --seed 42

# 4. Smoke test (200 examples)
.venv/bin/python train.py --config configs/cpmg_full.yaml --limit 200 --suffix _smoke

# 5. Train CPMG, content-only baseline, ablations, write-only, cross-source (about 1.5 h on the RTX 5060)
./run_all_training.sh            # or: .venv/bin/python train.py --config configs/<name>.yaml

# 6. Other baselines
.venv/bin/python -m baselines.keyword
.venv/bin/python -m baselines.tfidf      # TF-IDF+LR (memory / full input / w/o ASB) + provenance-only leak check
.venv/bin/python -m baselines.piguard    # downloads leolee99/PIGuard (~740 MB)

# 7. Evaluation: metrics, tables, plots
.venv/bin/python eval.py

# 8. Report
.venv/bin/python report/build_report.py   # -> report/report.pdf (+ report/report.tex)
```

Checkpoints are written to `checkpoints/` (git-ignored; about 0.7 GB each). Only two are published, on Hugging Face
(https://huggingface.co/ShahmeerHyat/cpmg-memory-poisoning): `cpmg_full/` (CPMG) and `content_only/` (content-only baseline). The ablation, write-only and
cross-source checkpoints (about 3.4 GB more) are not uploaded; rebuild them with `./run_all_training.sh`.

## Honest scope
- Single seed (42); no confidence intervals across seeds. AUROC 95% intervals are bootstrap over test examples.
- No MPBench (no data release). L2/L3 are derived from AgentPoison records, not collected in the wild.
- A-MemGuard, BlindGuard, end-to-end attack success, LongMemEval utility and multiple seeds are deferred to the paper (18 Nov).
- See `results/DECISIONS.md` for every data/method decision and `report/report.pdf` for limitations.
