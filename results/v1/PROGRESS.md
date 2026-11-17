# Progress

- 2026-11-07: Steps 1-2 done (env, sources, AgentPoison/ASB record production).
- 2026-11-07: Step 3 done: dataset built (`data/build_dataset.py`), splits in `data/processed/`, examples in `results/data_examples.md`.
- 2026-11-07: Step 4 done: smoke test (200 train examples) passed after two fixes (fp32 master weights; gradient checkpointing). `results/runs/cpmg_full_smoke.json`.
- 2026-11-07: Step 5 started: 7 DeBERTa runs via `run_all_training.sh`.
- 2026-11-07: cpmg_full trained (val AUROC in results/runs/cpmg_full.json). Baselines keyword/TF-IDF/prov-only done. eval.py, report builder and report draft (sections 1-4) written. Training of remaining 6 runs continues in background via run_all_training.sh.
- 2026-11-08: Step 5 done: all 7 DeBERTa runs trained (results/runs/*.json).
- 2026-11-08: Step 6 done: PIGuard zero-shot scored (keyword, TF-IDF, provenance-only done earlier).
- 2026-11-08: Step 7 done: eval.py -> results/metrics.csv/json, tables.md, policy.json, plots/.
- 2026-11-08: Step 8 done: report/report.pdf (5 pages, numbers filled from results/ by report/build_report.py), report/report.tex, README.md.
- 2026-11-08: Submission prep: DATA_LICENSES.md, HF model card, links in README/report, submission/CPMG_submission.zip, personal paths removed from logs.
- 2026-11-08: Published: GitHub https://github.com/ShahmeerHyat/cpmg-memory-poisoning (public), HF https://huggingface.co/ShahmeerHyat/cpmg-memory-poisoning (cpmg_full + content_only checkpoints).
