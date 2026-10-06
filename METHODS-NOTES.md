# Research provenance and evaluation notes

## Original paper and notebook

The original AP Research paper asks how effectively LightGBM can classify bankruptcy among U.S. small-cap companies. It intentionally prioritizes recall over precision. Its reported tuned-model counts are TP 18, TN 2,998, FP 137, FN 12; the saved notebook agrees. Recall rises from 50% to 60% compared with the baseline, a **10 percentage-point** increase, while precision falls.

The PDF is included unchanged. The original notebook copy preserves the statistical workflow and saved model results, with presentation changes: headings, a portable CSV path, local warning text removed, and the unsuccessful optional Graphviz cell omitted. The original notebook's saved execution order is inconsistent, so the output is retained as historical evidence rather than claimed as reproducible from a clean execution.

## Code–paper distinctions

- **Test years:** the code trains through 2015 and tests on 2016–2018. Page 15 describes a 2015–2018 test period. The repository documents the implemented split.
- **Company selection:** the code filters companies by their median X8 across their recorded history, retaining all rows of those companies. Page 11's wording can be read as filtering each row by its own market value.
- **Target:** page 13 refers to the actual bankruptcy-filing year. The code selects the last observed row; no filing-date column is supplied. This is an assumption, not a verified event date.
- **Cross-validation:** SMOTE is applied before cross-validation, and the row sequence is company-first rather than globally ordered by year. Calling TimeSeriesSplit does not make that sequence chronological. Some saved outputs call the score F1 although the current source requests recall.
- **Feature importance:** early models explicitly use gain importance; later baseline/tuned models use the default split-count importance while their plots are labelled as gain. The extension uses gain explicitly.
- **AUC:** the historical baseline uses hard class predictions in its ROC calculation. The extension uses prediction scores.
- **Dataset duration:** 1999–2018 spans 19 elapsed years and includes 20 calendar-year values. The repository uses the date range to avoid ambiguity.

## Separate evaluation extension

`train.py` and `validation_revision.ipynb` were prepared during repository cleanup. They use five explicit expanding calendar windows, train-only company selection, fold-local scaling and SMOTE, all 18 features, seeded Optuna, and a fixed decision threshold. A baseline and tuned candidate are compared using only development-fold scores; only the selected candidate is evaluated on the holdout in this run.

The extension optimizes **average precision**, rather than the original study's recall objective, to examine ranking performance across thresholds. It is an additional experiment, not a replacement for the research question or original reported scores. Its different cohort and preprocessing mean the two runs are not a controlled before/after performance comparison.

Year separation and fold-local resampling address specific implementation problems. They do not establish verified label availability, reporting dates, an actual forecast horizon, generalization to unseen companies, or investment utility. The original holdout period was already inspected in the original project and is not an untouched external validation dataset.

## Validation completed

- Full 30-trial extension run completed in the author's existing Python environment.
- Seven protocol tests passed: calendar separation, future-data cohort invariance, fit-only resampling, score-based AUC, single-class rejection, order-independent terminal labels, and duplicate company/year rejection.
- Final source was checked against the winning development scores and the selected model's full holdout prediction vector.
- The notebook's code cells were syntax checked. The command-line extension was executed; neither notebook interface was executed end-to-end.
- The original saved confusion matrix matches the paper's Table 3 and Figures 11–12. New result figures were visually inspected.
- A scoped scan found no common token/private-key patterns or local user paths in packaged source and documentation. This is not an exhaustive security audit.
- The raw CSV and unrelated practice script are omitted. No license for the author's code has been selected, and nothing has been published by this packaging process.
