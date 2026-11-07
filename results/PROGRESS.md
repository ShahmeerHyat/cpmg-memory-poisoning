# Progress

- 2026-11-07: Steps 1-2 done (env, sources, AgentPoison/ASB record production).
- 2026-11-07: Step 3 done: dataset built (`data/build_dataset.py`), splits in `data/processed/`, examples in `results/data_examples.md`.
- 2026-11-07: Step 4 done: smoke test (200 train examples) passed after two fixes (fp32 master weights; gradient checkpointing). `results/runs/cpmg_full_smoke.json`.
- 2026-11-07: Step 5 started: 7 DeBERTa runs via `run_all_training.sh`.
- 2026-11-07: cpmg_full trained (val AUROC in results/runs/cpmg_full.json). Baselines keyword/TF-IDF/prov-only done. eval.py, report builder and report draft (sections 1-4) written. Training of remaining 6 runs continues in background via run_all_training.sh.
- PAUSED (usage limit). Remaining: wait for run_all_training.sh to finish -> python -m baselines.piguard -> python eval.py -> write results/limitations sections in report/report.md (replace RESULTS_PLACEHOLDER) -> python report/build_report.py -> commit.
- 2026-11-08: Step 5 done: all 7 DeBERTa runs trained (results/runs/*.json).
