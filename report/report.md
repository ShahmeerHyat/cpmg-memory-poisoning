# CPMG: Contextual Provenance Memory Guard — Methodology and First Results

<div class="cover">
Agentic AI (AI-4015) · FAST-NUCES Islamabad · Instructor: Sohail Abbas<br>
Shahmeer Ali (23I-0119) · Mohammad Soban (23I-0056) · Ali Rauf (23I-0076)<br>
Deliverable: Proposed Methodology and Implementation · 8 November 2026
</div>

## 1 Problem and threat model

LLM agents save content from email, chat, documents, web pages and tool outputs into a persistent memory and
reuse it in later sessions. An attacker who controls any of that content can plant a record now that changes an
answer or a tool call later. We consider three attack levels. **L1 (direct):** one stored record carries the whole
malicious instruction. **L2 (compositional):** the harmful content is spread over two or more records, and it only
takes effect when they are retrieved together. **L3 (dormant):** the record does nothing until a later query (or
entity, date or action) supplies its trigger. The attacker controls record *content* and can choose wording; they
cannot edit system-recorded metadata (which channel a record came through, its authority tier, its parent tool
call). A pointwise filter such as PIGuard scores one string at a time, so by construction it cannot see the
neighbour that completes an L2 attack or the query that wakes an L3 record. A defence also has to keep useful
memory, so we report false rejections of matched benign records alongside detection.

## 2 Method

![CPMG pipeline. The two CPMG checkpoints share one model family; the authority policy is a separate deterministic module.](figures/cpmg_diagram.png)

**Provenance envelope.** Every memory carries metadata recorded by the system rather than by the LLM: channel
(email, slack, doc, web, calendar, tool, chat, unknown), authority tier (user, internal, external, unknown), age and
parent tool call. Each value is a new special token added to the tokenizer.

**Model.** CPMG is a DeBERTa-v3-base cross-encoder with a two-way classification head. Its input is
`[CLS] provenance-tokens memory [SEP] source-interaction [SEP] query/action [SEP] nbr1 [SEP] nbr2 [SEP] nbr3 [SEP]`
(maximum 512 tokens; budgets of 32/48/64 tokens for source, query and each neighbour; the memory gets the rest).
The risk score is P(poisoned).

**Two checkpoints.** At *write* time there is no future query yet, so the write model sees provenance, memory and
source only. At *retrieval* time CPMG re-scores the record with the current query or planned action and the
co-retrieved memories. "Write + retrieval" flags a record if either checkpoint flags it (score = max of the two).

**Training.** Weighted cross-entropy (class weights inverse to frequency), AdamW, learning rate [[run:cpmg_full:config:lr]], linear schedule
with warm-up, [[run:cpmg_full:config:epochs]] epochs, batch [[run:cpmg_full:config:batch_size]] with gradient accumulation [[run:cpmg_full:config:grad_accum]], bf16 autocast over fp32 weights, gradient
checkpointing, seed 42. The checkpoint with the best validation AUROC is kept. The supervised contrastive term was
not added (time).

**Authority policy.** A small deterministic module, separate from the model: memory whose authority is `external`
or `unknown` can never authorise a protected action (payment, send/share data, change settings, delete). It does
not use the risk score.

## 3 Data and splits

**Real public data only.** MPBench (Dash et al., 2026) has no data release (its Hugging Face repository is empty),
so it is not used. We use five public sources: **AgentPoison** (NeurIPS 2024; StrategyQA and EHRAgent memories),
**ASB** (ICLR 2025; memory-poisoning plans from its direct-prompt-injection runs), **MemoryGraft** (benign and poisoned
experience records), **memshield-bench** (whose poisoned entries were written by the dataset's author) and
**NotInject** ([[stats:test_notinject:n]] benign prompts containing trigger words, used only to test over-defence). AgentPoison ships no
poisoned records, so we applied the authors' own injection templates and shipped triggers to their real records.
Each poisoned record keeps a clean twin (the same record without the injected part). ASB ships poisoned memory only,
so we ran ASB's clean planning path with a local Qwen3-8B model to get clean plans for the same agents and tasks.
Provenance comes from each source's documented origin (Table 1 in `results/DECISIONS.md`, D4), because no source
records per-item metadata.

**Derived L2 and L3.** L1 cases are real records as stored. L2 and L3 are **derived** from AgentPoison records, where
the injected span and trigger are known exactly. *L2:* the scored memory is the record without its injected span
(identical to its clean twin), and the injected span is stored as a separate co-retrieved record; the negative is the
same memory with clean neighbours only. *L3:* the scored memory is the full poisoned record, whose backdoor waits for
a trigger; the positive has the trigger in the later query, and the negative is the same question without it. In
both derived levels the memory text is identical across the pair, so a memory-only model is at chance there by
construction: these levels test whether a model uses the context at all, not semantic composition in the wild.

**Splits.** Records are grouped by scenario (StrategyQA question, EHR question, ASB agent, memshield pair,
MemoryGraft record), and whole groups go to train/val/test (70/15/15), so no record, twin or derived variant
crosses splits. The **BadChain trigger family is held out**: its groups never appear in train or val. ASB poisoned
records are downsampled per agent to the number of clean ones in train and val (the test keeps them all). As a
**cross-source test**, a second CPMG model is trained without ASB and tested on all ASB records. Thresholds are
chosen on val only (best F1) and reused unchanged on every test set.

<p class="tcap">Table 1. Dataset (seed 42). Background clean = StrategyQA knowledge-base paragraphs.</p>

[[table:data]]

## 4 Baselines

**Content-only DeBERTa:** the same model and training, memory text only (the pointwise baseline).
**PIGuard:** the public checkpoint `leolee99/PIGuard`, zero-shot on the memory text; we rebuilt its model class (a
DeBERTa-v2 encoder with a linear head) locally and loaded the published weights. **Keyword filter:** a fixed list of [[run:keyword:n_patterns]] generic injection patterns, no
training. **TF-IDF + logistic regression:** on the memory text, and on the same flattened input CPMG sees. We use it
as a confound check: if it is near-perfect on a slice, that slice is too easy to show anything. **Provenance-only
logistic regression:** one-hot channel, authority and parent only, to check that provenance does not leak the label.
**Ablations:** CPMG without provenance, without query/action, without neighbours, write-only, and write + retrieval.

RESULTS_PLACEHOLDER
