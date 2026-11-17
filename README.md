# CPMG — Contextual Provenance Memory Guard

Course project, Agentic AI (AI-4015), FAST-NUCES Islamabad.
Group: Shahmeer Ali (23I-0119), Mohammad Soban (23I-0056), Ali Rauf (23I-0076).

- Code and results: https://github.com/ShahmeerHyat/cpmg-memory-poisoning
- Model checkpoints: https://huggingface.co/ShahmeerHyat/cpmg-memory-poisoning
- **Paper: `paper/paper.pdf`** (ACL format; source `paper/paper.tex.in`; anonymous review version `paper/paper_review.pdf`) · Data licenses: `DATA_LICENSES.md`
- Earlier course report (8 Nov, dataset v1): `report/report.pdf`. Its numbers are superseded; see "Versions" below.

CPMG is a DeBERTa-v3-base cross-encoder that reads a stored agent memory together with its source interaction, the
current query/action, 3 co-retrieved memories and system-recorded provenance tokens, and gives **two scores**:
*poison* (attacker-written content is in the context) and *activation* (the query carries the trigger that content
is conditional on). A separate deterministic authority policy stops low-authority memory from authorising protected
actions. Data come from public sources (`data/raw/SOURCES.md`); L2/L3 cases are **derived** by transforming real
records. Generated content is limited to: random training triggers (test triggers are the real AgentPoison ones),
ASB clean plans (local Qwen3-8B), and the reworded test attacks of the paraphrase experiment.

## Layout
```
data/sources/      record production from the raw sources (AgentPoison templates, ASB export, ASB clean run),
                   paraphrase_attack.py (reworded test attacks)
data/build_dataset.py   dataset v2: matched cases, triggers, grouped splits -> data/processed/*.jsonl
cpmg/              features.py (input formatting, provenance tokens), model.py (two outputs), policy.py
baselines/         content_only.py, piguard.py, keyword.py, tfidf.py (+ provenance-only leak check),
                   lexmatch.py (copy rule and quoted-copy rule)
train.py           training, resumable; start_training.sh / stop_training.sh / run_all_training.sh
score_split.py     score extra test files with trained checkpoints (no training)
eval.py            all metrics, tables and plots from prediction files
configs/*.yaml     one config per DeBERTa run
results/           metrics.csv/json, tables.md, plots/, preds/, runs/, logs/, DECISIONS.md, PROGRESS.md, v1/
paper/             paper.tex.in (template), build_paper.py (fills every number, compiles), make_figures.py,
                   figs/*.tex (TikZ diagrams), paper.pdf
report/            8 Nov course report (dataset v1)
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

# 3. Dataset and splits (v2)
.venv/bin/python data/build_dataset.py --seed 42

# 4. Smoke test (200 examples)
.venv/bin/python train.py --config configs/cpmg_full.yaml --limit 200 --suffix _smoke

# 5. Train: 8 configs on seed 42, then seeds 43 and 44 for CPMG, content-only and CPMG w/o provenance
#    (14 runs, about 6 h on the RTX 5060). Resumable: stop any time, start again later.
./start_training.sh              # runs in the background; progress: tail -f results/logs/train_all.nohup
./stop_training.sh               # saves state and stops; ./start_training.sh continues from there

# 6. Other baselines
.venv/bin/python -m baselines.keyword
.venv/bin/python -m baselines.lexmatch   # copy rule + quoted-copy rule
.venv/bin/python -m baselines.tfidf      # TF-IDF+LR (memory / full input / w/o ASB) + provenance-only leak check
.venv/bin/python -m baselines.piguard    # downloads leolee99/PIGuard (~740 MB)

# 7. Paraphrase attack (needs the local LLM server from step 2; stop it before scoring)
.venv/bin/python data/sources/paraphrase_attack.py --url http://127.0.0.1:8091   # -> data/processed/test@para*.jsonl
.venv/bin/python score_split.py --splits test@para test@paranq --models cpmg_full cpmg_full_s43 cpmg_full_s44 \
    abl_no_prov abl_no_prov_s43 abl_no_prov_s44 content_only content_only_s43 content_only_s44 \
    cpmg_noehr abl_no_query abl_no_nbrs write_only
#    then rerun step 6 so the baselines are scored on the reworded files too
#    (data/interim/paraphrases.jsonl and the two test@para*.jsonl files are committed, so generation can be skipped)

# 8. Evaluation: metrics, tables, plots
.venv/bin/python eval.py

# 9. Paper (fills every number from results/, draws figures, compiles)
.venv/bin/python paper/build_paper.py --review   # -> paper/paper.pdf and the anonymous paper/paper_review.pdf
#    Compiles with tectonic or latexmk if installed, otherwise with the Docker image texlive/texlive:latest-full.
```

Checkpoints are written to `checkpoints/` (git-ignored; about 0.7 GB each). Two are published on Hugging Face
(https://huggingface.co/ShahmeerHyat/cpmg-memory-poisoning): `cpmg_full/` (CPMG, seed 42) and `content_only/` (content-only
baseline, seed 42), both from dataset v2 and matching the paper. The earlier v1 weights remain in that repository's commit
history. The other checkpoints are not uploaded; rebuild them with `./start_training.sh`.
Model card: `.venv/bin/python hf/make_model_card.py` (add `--upload` to publish).

## Versions
- **v2 (17 Nov, the paper).** Two outputs; per-record random training triggers; the five real AgentPoison triggers are
  test-only; wrong-trigger and benign-overlap negatives; 3 seeds for the main models; paraphrase attack; cross-template run.
- **v1 (8 Nov, course report; git tag `v1-nov8`).** Single risk label. Every L3 negative was the same input as an L1 positive with the opposite
  label, so the v1 L3 and all-levels numbers are not valid (DECISIONS D22). Results kept in `results/v1/`; the v1 data splits are in the tagged commit.

## Honest scope
- Three seeds for CPMG, content-only and CPMG w/o provenance; one seed for everything else. The dataset is built once.
- All L2/L3 cases derive from AgentPoison (one injection sentence per agent). Training triggers are synthetic; only five
  real triggers exist for testing. The paraphrase attack uses one rewriting model and is not optimised against CPMG.
- Provenance is constant per source, so its value cannot be tested on this data; the model ignores the provenance tokens.
- Not done: A-MemGuard, adapted BlindGuard, end-to-end attack success in an agent, LongMemEval utility, contrastive term.
- No MPBench (no data release).
- See `results/DECISIONS.md` for every data/method decision and the paper's Limitations section.
