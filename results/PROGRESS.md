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
- 2026-11-16: v1 archived (`results/v1/`, `data/processed_v1/`, `checkpoints/v1/`). Found the v1 label conflict (D22). Dataset v2 built: two targets, per-record random training triggers, real triggers test-only, wrong-trigger and benign-overlap negatives (D23, D24, D29). train.py: two outputs, provenance dropout, seeds, resumable checkpoints (`start_training.sh` / `stop_training.sh`).
- 2026-11-17: All 14 runs trained (7 configs + cross-template on seed 42; seeds 43, 44 for CPMG, content-only, CPMG w/o provenance). Baselines re-run (keyword, TF-IDF, PIGuard, copy rule, quoted-copy rule, provenance-only). Paraphrase attack generated and scored (D31). eval.py -> results/metrics.csv/json, tables.md, plots/. Paper: `paper/paper.pdf` (ACL format, 10 pages incl. references and appendix), numbers filled by `paper/build_paper.py`.
- Not done (stated in the paper): A-MemGuard, adapted BlindGuard, end-to-end attack success in an agent, LongMemEval utility, supervised contrastive term.
- 2026-11-17: References completed, cross-template wording corrected, anonymous review PDF added (`paper/paper_review.pdf`). v2 checkpoints (cpmg_full, content_only; seed 42) and model card uploaded to Hugging Face. Git tag `v1-nov8` marks the 8 Nov submission.
