# CPMG — Agentic AI Course Project

Read this whole file before doing anything. Then read everything in `docs/`.

## Who / what
- Course: Agentic AI (AI-4015), FAST-NUCES Islamabad. Instructor: Sohail Abbas.
- Group: Shahmeer Ali (23I-0119), Mohammad Soban (23I-0056), Ali Rauf (23I-0076).
- Project: **CPMG — Contextual Provenance Memory Guard**, a defense against persistent memory poisoning in LLM agents.

## Deadlines
1. **Sun 8 Nov 2026, 11:59 PM (PKT) — "Proposed Methodology and Implementation."**
   Brief: "Train and Test your proposed methodologies for the project. Submission requires trained and evaluated code and results."
   This is the current priority.
2. **Wed 18 Nov 2026 — "Final Results and Paper Writeup."** PDF paper in LaTeX or Word, formatted like the target conference (plan: ACL/ARR template).

## Background docs (in `docs/`)
- `agentic_lit_review.pdf`: the literature review + brief proposal we submitted to THIS course. **This is the proposal we are graded against.**
- `mbr_proposal.pdf`: longer proposal of the same project, written for another course (more detail on method and evaluation).
- `mbr_a2_systematic_review.pdf`: systematic review (25 peer-reviewed studies), written for another course. Use it for related work and the evaluation design, not as a deliverable.

Do not copy text from the other-course docs verbatim into deliverables. Rewrite it.

## The problem
Agents store content from email, Slack, documents, web pages and tool outputs in persistent memory and reuse it later. An attacker can plant content that is stored now and changes an answer or tool call later. Hard cases:
- **L1 direct**: one record holds the whole malicious instruction.
- **L2 compositional**: the harm is split across 2+ records that each look benign; it activates only when they are retrieved together.
- **L3 dormant**: the record looks harmless until a later query, entity, date or action triggers it.

Pointwise filters (e.g. PIGuard) score each record alone, so they miss L2/L3. A-MemGuard uses context but needs several LLM calls per check.

## The method (CPMG)
- **Provenance envelope**: each memory carries system-recorded metadata the LLM cannot edit: channel (email/slack/doc/web/calendar/tool), source authority tier (user / internal / external / unknown), age, parent tool call.
- **Model**: DeBERTa-v3-base cross-encoder. Input = `[provenance tokens] memory [SEP] source interaction [SEP] current query or planned action [SEP] up to 5 related memories`. Add the provenance values as special tokens. Output: risk score.
- **Two checkpoints**: at **write** (memory + source + provenance; no future query yet) and at **retrieval** (rescore with the query/action and co-retrieved neighbours).
- **Training**: weighted cross-entropy. Add a supervised contrastive term (attack vs its benign twin) if time allows.
- **Authority policy** (deterministic, separate from the model): memory from an external/unknown source can never authorise a protected action (payment, send/share data, change settings, delete). Small module, report how many protected actions it blocks.

## Data
**No invented or template data at all.** Real public datasets only.
1. **Sources**:
   - AgentPoison (github BillChan226/AgentPoison)
   - ASB (github agiresearch/ASB; memory-poisoning cases)
   - MemoryGraft (github Jacobhhy/Agent-Memory-Poisoning)
   - memshield-bench (HF npow/memshield-bench). Note: its poisoned entries were written by the dataset author.
   - PIGuard NotInject, if public.
   - MPBench (Dash et al., arXiv:2606.04329): check once. No data release found so far; drop it if there isn't one.
2. **L2/L3 only by transforming real records**: split a real poisoned record into fragments (L2), or move its trigger into the later query (L3). Each derived example keeps its `source_dataset` and `source_id`. The report calls this split **"derived"**.
3. **Provenance must not leak the label**: clean records also come from external sources. Train a provenance-only classifier as a check and report its score.
4. **If the public data isn't enough, stop and tell me.** Never fill gaps with made-up data.
5. **Approved 7 Nov 2026** (after the source check showed one-label-per-source confounds):
   - A: produce AgentPoison poisoned records with the authors' own injection template and shipped trigger.
   - B: run ASB's clean (no-attack) agent with a local llama.cpp model to get clean memory for the same agents/tasks.
   - C: provenance tags come from each source's documented origin. Run the provenance-only leak check; if provenance only encodes the dataset, report it as untestable here.
6. **Splits**: grouped by source record / scenario, so no record (or its derived L2/L3 variants) or entity crosses train/val/test. Hold out **at least one whole attack family** from training. If cheap, also hold out one channel. Choose thresholds on val only.

## Baselines and ablations
- **Content-only DeBERTa**: same model, memory text only. This is the pointwise baseline.
- **PIGuard**: public checkpoint if available on Hugging Face, zero-shot. Verify it exists; don't assume.
- **Keyword/regex filter**: trivial baseline.
- **Ablations**: CPMG without provenance, without query/action, without neighbours. Write-time only vs write + retrieval.
- **Optional if time**:
  - a paraphrase adaptive attack (rewrite test attacks, re-score)
  - forged-reasoning-trace attack family
- **Deferred to the paper (Nov 18)**: A-MemGuard (LLM-based; small subset via a local llama.cpp model), adapted BlindGuard, end-to-end attack success in a sandboxed agent, LongMemEval utility, 3 seeds.

## Metrics
- AUROC
- TPR at 1% and 5% FPR
- F1 at the chosen threshold
- False-rejection rate on benign twins
- **All of the above broken down by L1/L2/L3 and for the held-out family**
- Latency per check (ms) and parameter count, vs baselines

## Hardware (IMPORTANT)
- Ubuntu laptop, **RTX 5060 (Blackwell), 8 GB VRAM**, ~15 GB system RAM, 512 GB NVMe.
- Blackwell needs a **PyTorch build with CUDA 12.8 (cu128)**. Check with `torch.cuda.is_available()` and a tiny matmul before anything else.
- Use **bf16, not fp16** (DeBERTa-v3 can overflow in fp16).
  - Start at batch 8 and max_len 512, and use grad accumulation to scale the effective batch.
  - Turn on gradient checkpointing if it OOMs.
- Make sure no other GPU process (e.g. llama.cpp servers) is running before training.
- Low system RAM: stream or tokenise in chunks and keep dataloader workers small (≤2). Running out of RAM swaps the machine to death.
- Target total training time: under ~2 hours for all runs. Use 2–3 epochs. Use a smaller max_len if inputs are short.

## Repo layout
```
data/        build_dataset.py, raw/, processed/ (splits as jsonl)
cpmg/        model.py, features.py (input formatting), policy.py
baselines/   content_only.py, piguard.py, keyword.py
train.py     eval.py     configs/*.yaml
results/     metrics.csv/json, plots, logs
report/      methodology + results report (LaTeX → PDF)
README.md    exact commands to reproduce everything
requirements.txt
```
- Fixed seed everywhere.
- Log every run's config and metrics to `results/`.
- Save checkpoints under `checkpoints/` and add it to .gitignore (too large for git; will be uploaded to Drive/HF).

## Deliverables for Nov 8
1. **Code**: the repo above, running end to end from README commands.
2. **Results**: metrics tables (CSV + markdown), ROC curves, training-loss plot, logs.
3. **Report (3–5 pages, PDF)**:
   - problem and threat model
   - CPMG method, with a diagram if possible
   - data and splits
   - baselines
   - results table
   - ablation table
   - per-level analysis
   - limitations and next steps
   - Cover info: course, group members, date.

## Rules
- **Never invent numbers.** Every number in the report must come from a file in `results/` produced by our code. If something didn't run, say so.
- Be honest about synthetic data, single seed, and what is deferred.
- Before each big step, state the plan in 2–3 lines. Then do it. Don't wait for approval on routine steps.
- Ask me before:
  - deleting anything outside this folder
  - any download over ~2 GB
  - changing the method in a way that contradicts the proposal
- Keep my messages short. I prefer direct, concise updates.

## Order of work
1. Environment check: GPU, torch cu128, bf16.
2. Download and inspect all data sources (see Data). Report counts, fields and licenses.
3. Build the dataset and splits. Sanity-check the counts and look at 5 examples per level.
4. Smoke test: tiny train/eval on 200 examples.
5. Full training: CPMG, content-only, ablations.
6. Baselines: PIGuard, keyword.
7. Evaluation, plots, tables.
8. Report PDF and README.
