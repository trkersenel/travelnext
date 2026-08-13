# Notebooks

This directory is for **exploration only**. Nothing here is part of the
pipeline, and nothing here is imported by `src/`, `api/` or `app/`.

The rule the project follows is that anything which produces a number quoted
anywhere — in the README, in a report, in a decision — must live in `src/` and
be reachable from a command:

```bash
python -m src.data.build_destinations    # ingestion
python -m src.data.build_dataset         # features + synthetic interactions
python -m src.analysis.eda               # figures + reports/eda_summary.md
python -m src.evaluation.run_experiments # the model comparison
python -m src.analysis.report_results    # reports/RESULTS.md
```

That way a reviewer can reproduce every claim without opening a notebook, and a
stale cell can never quietly contradict the reported results.

The exploratory analysis that would normally live in a notebook is in
[`src/analysis/eda.py`](../src/analysis/eda.py), which writes its figures to
`reports/figures/` and its summary to `reports/eda_summary.md`.
