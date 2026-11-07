# Raw data sources (downloaded 2026-11-07)

| Source | Origin | Version | License | Poisoned | Clean |
|---|---|---|---|---|---|
| ASB | github agiresearch/ASB | 544540ff0788998072f7df463056fd17e62cc9f4 | MIT | 2,012 unique memory records (`memory_db/direct_prompt_injection/*/chroma.sqlite3`; 400 attacker instructions x 5 DPI styles, 10 agents) | 0 in memory DBs |
| AgentPoison | github AI-secure/AgentPoison (was BillChan226/AgentPoison) | 7b02e28f5e477052bd16c778fef7c1f78f7c9d7c | MIT | 0 shipped (built at runtime in `ReAct/local_wikienv.py`) | 9,251 StrategyQA paragraphs; 317 EHR questions; 199 EHR logs |
| MemoryGraft | github Jacobhhy/MemoryGraft (was Jacobhhy/Agent-Memory-Poisoning) | 823662058d1e0fe49306d1820d7947297ef3c252 | MIT | 10 (`memorygraft/payloads/experience_seeds.json`) | 100 |
| memshield-bench | HF npow/memshield-bench | v1.0.0 | CC-BY-SA-4.0 | 322 (written by the dataset author) | 856 (500 copied verbatim from AgentPoison StrategyQA, 199 from AgentPoison EHR) |
| NotInject | HF leolee99/NotInject | - | MIT | 0 | 339 benign hard negatives (train/val/test files duplicate subsets one/two/three) |
| MPBench | arXiv:2606.04329 | - | - | No data release. HF `pritamdash/mp-bench` is empty (only .gitattributes). Dropped. | |

Shallow clones (`git clone --depth 1`). No code from these folders has been executed.

Note: `ASB/data/agent_task_pot_all.jsonl` in the ASB repo is a committed GitHub rate-limit HTML page, not data. Excluded.

## Records produced from these sources (approved 7 Nov 2026; scripts in `data/sources/`)

| Output | Script | How | Records |
|---|---|---|---|
| `data/interim/agentpoison.jsonl` | `agentpoison_inject.py` | Authors' injection templates (`ReAct/local_wikienv.py:188-197`, `EhrAgent/ehragent/medagent.py:117-128`) applied to their shipped StrategyQA train questions and EHR memory; shipped triggers (optimised AP trigger, BadChain). Clean twin = same record without the injected part. No LLM. | StrategyQA 2,290 poisoned + 2,290 clean twins; EHR 193 + 193 |
| `data/interim/asb_poisoned.jsonl` | `asb_export_poisoned.py` | Read-only export of ASB's shipped DPI memory DBs (gpt-4o-mini) | 2,012 poisoned (10 agents, 1 task each, 400 attacker tools, 5 injection styles) |
| `data/interim/asb_clean.jsonl` | `asb_clean_run.py` | ASB clean planning path run locally with Qwen3-8B-Q4_K_M (llama.cpp), 5 samples/task | 250 clean (51 tasks x 5 samples; 5 samples failed ASB's plan check after 10 attempts) |
