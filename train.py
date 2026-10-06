"""Chronological LightGBM evaluation for a retrospective bankruptcy-label proxy."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import importlib.metadata

import numpy as np
import pandas as pd
import optuna
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)

FEATURES = [f"X{i}" for i in range(1, 19)]
SEED = 42
# Expanding windows; no calendar year crosses a train/validation boundary.
VALIDATION_WINDOWS = [(2006, 2007), (2008, 2009), (2010, 2011), (2012, 2013), (2014, 2015)]


def load_data(path):
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = ["company_name", "status_label", "year", *FEATURES]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = frame[required].copy()
    if frame.isna().any().any():
        raise ValueError("Missing values must be investigated before fitting.")
    if not frame.status_label.isin(["alive", "failed"]).all():
        raise ValueError("Expected alive/failed labels.")
    if not np.isfinite(frame[["year", *FEATURES]].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite numeric values found.")
    if not np.equal(frame.year, np.floor(frame.year)).all():
        raise ValueError("Years must be whole numbers.")
    frame["year"] = frame.year.astype(int)
    if frame.duplicated(["company_name", "year"]).any():
        raise ValueError("Duplicate company/year records found.")
    if (frame.groupby("company_name").status_label.nunique() != 1).any():
        raise ValueError("This target definition requires a constant company-level status.")
    frame = frame.sort_values(["company_name", "year"], kind="stable").reset_index(drop=True)
    last_year = frame.groupby("company_name").year.transform("max")
    # Preserve the original project's retrospective label assumption, explicitly.
    # There is no independent bankruptcy-event date or reporting-availability date.
    frame["target"] = ((frame.status_label == "failed") & (frame.year == last_year)).astype(int)
    return frame


def select_partitions(frame, cutoff, end_year, lower=250.0, upper=2000.0):
    """Select a cohort using only history up to the training cutoff."""
    history = frame.loc[frame.year <= cutoff]
    median = history.groupby("company_name").X8.median()
    companies = median.index[(median > lower) & (median < upper)]
    eligible = frame.company_name.isin(companies)
    train = frame.loc[eligible & (frame.year <= cutoff)]
    validation = frame.loc[eligible & (frame.year > cutoff) & (frame.year <= end_year)]
    train = train.sort_values(["year", "company_name"]).reset_index(drop=True)
    validation = validation.sort_values(["year", "company_name"]).reset_index(drop=True)
    if train.empty or validation.empty:
        raise ValueError(f"Empty partition at cutoff {cutoff}.")
    if train.year.max() >= validation.year.min():
        raise AssertionError("Training years must precede validation years.")
    if train.target.nunique() != 2 or validation.target.nunique() != 2:
        raise ValueError(f"Both classes are required in each partition at cutoff {cutoff}.")
    return train, validation


def make_pipeline(params, minority_count):
    if minority_count < 2:
        raise ValueError("SMOTE needs at least two minority training examples.")
    return Pipeline([
        # Fit scaling inside each fold: SMOTE's distances depend on feature units.
        ("scale", StandardScaler().set_output(transform="pandas")),
        ("smote", SMOTE(random_state=SEED, k_neighbors=min(5, minority_count - 1))),
        ("model", LGBMClassifier(
            random_state=SEED, n_jobs=1, verbosity=-1, importance_type="gain",
            deterministic=True, force_col_wise=True, **params,
        )),
    ])


def fit_model(train, params):
    model = make_pipeline(params, int(train.target.value_counts().min()))
    model.fit(train[FEATURES], train.target)
    return model


def score_predictions(y_true, probabilities, threshold=0.5):
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "threshold": threshold,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def cross_validate(frame, params):
    scores = []
    for start, end in VALIDATION_WINDOWS:
        train, valid = select_partitions(frame, start - 1, end)
        model = fit_model(train, params)
        probabilities = model.predict_proba(valid[FEATURES])[:, 1]
        scores.append({
            "training_through": start - 1, "validation_start": start, "validation_end": end,
            "training_rows": len(train), "training_positives": int(train.target.sum()),
            "validation_rows": len(valid), "validation_positives": int(valid.target.sum()),
            "average_precision": float(average_precision_score(valid.target, probabilities)),
        })
    return scores


def save_plots(y_true, probabilities, model, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    ConfusionMatrixDisplay.from_predictions(
        y_true, probabilities >= 0.5, display_labels=["Other records", "Terminal failure"],
        ax=axes[0], colorbar=False, cmap="Blues",
    )
    axes[0].set_title("2016–2018 holdout · threshold 0.5")
    PrecisionRecallDisplay.from_predictions(y_true, probabilities, ax=axes[1], name="LightGBM")
    axes[1].axhline(float(np.mean(y_true)), color="gray", linestyle="--", label="Positive prevalence")
    axes[1].set_title("Precision–recall tradeoff")
    axes[1].legend(fontsize=8)
    fig.savefig(destination / "evaluation.png", bbox_inches="tight")
    plt.close(fig)
    gains = model.named_steps["model"].feature_importances_
    order = np.argsort(gains)
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    ax.barh(np.array(FEATURES)[order], gains[order], color="#2563eb")
    ax.set(xlabel="Total split gain", title="LightGBM feature importance (training data)")
    fig.savefig(destination / "feature_importance.png", bbox_inches="tight")
    plt.close(fig)


def run_experiment(data_path, output_dir, trials=30):
    if trials < 1:
        raise ValueError("At least one tuning trial is required.")
    path, output = Path(data_path), Path(output_dir)
    frame = load_data(path)
    development = frame.loc[frame.year <= 2015].copy()
    baseline_params = {"n_estimators": 100, "learning_rate": 0.1}
    baseline_cv = cross_validate(development, baseline_params)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 5, 10),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        }
        folds = cross_validate(development, params)
        trial.set_user_attr("folds", folds)
        return float(np.mean([fold["average_precision"] for fold in folds]))

    def report(study, trial):
        if (trial.number + 1) % 5 == 0 or trial.number == 0:
            print(f"Tuning: {trial.number + 1}/{trials} trials; best mean validation AP {study.best_value:.4f}", flush=True)

    study.optimize(objective, n_trials=trials, callbacks=[report])
    baseline_ap = float(np.mean([fold["average_precision"] for fold in baseline_cv]))
    # Select solely on development folds; retain the baseline if tuning is worse.
    use_tuned = study.best_value >= baseline_ap
    selected_params = study.best_params if use_tuned else baseline_params
    train, test = select_partitions(frame, 2015, 2018)
    model = fit_model(train, selected_params)
    probabilities = model.predict_proba(test[FEATURES])[:, 1]
    results = score_predictions(test.target, probabilities)
    versions = {name: importlib.metadata.version(name) for name in
                ["numpy", "pandas", "lightgbm", "scikit-learn", "imbalanced-learn", "optuna", "matplotlib"]}
    metrics = {
        "experiment": "chronological_revision_v1", "seed": SEED, "trials": trials,
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "python_version": platform.python_version(), "package_versions": versions,
        "target": "last observed record of a company labelled failed; retrospective proxy",
        "cohort": "250 < median X8 < 2000, computed using years up to each training cutoff",
        "cohort_bounds": {"lower_exclusive": 250, "upper_exclusive": 2000},
        "features": FEATURES, "data_rows": len(frame), "data_companies": int(frame.company_name.nunique()),
        "year_min": int(frame.year.min()), "year_max": int(frame.year.max()),
        "training_rows": len(train), "training_positives": int(train.target.sum()),
        "smote_training_rows_per_class": int(train.target.value_counts().max()),
        "test_rows": len(test), "test_positives": int(test.target.sum()),
        "test_companies": int(test.company_name.nunique()),
        "test_positive_prevalence": float(test.target.mean()),
        "all_negative_accuracy": float(1 - test.target.mean()),
        "baseline_cv": baseline_cv, "baseline_mean_cv_ap": baseline_ap,
        "best_tuned_mean_cv_ap": study.best_value, "best_tuned_cv": study.best_trial.user_attrs["folds"],
        "best_tuned_params": study.best_params,
        "selected_model": "tuned" if use_tuned else "baseline", "selected_params": selected_params,
        "holdout": results,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    study.trials_dataframe().drop(columns=["user_attrs_folds"], errors="ignore").to_csv(output / "trials.csv", index=False)
    predictions = test[["company_name", "year", "target"]].copy()
    predictions["probability"] = probabilities
    predictions["prediction"] = (probabilities >= 0.5).astype(int)
    predictions.to_csv(output / "predictions.csv", index=False)
    model.named_steps["model"].booster_.save_model(str(output / "lightgbm.txt"))
    # Model-only weights require these transformations for equivalent inference.
    np.savez(output / "scaler.npz", mean=model.named_steps["scale"].mean_, scale=model.named_steps["scale"].scale_)
    save_plots(test.target, probabilities, model, output)
    print(json.dumps({"selected_model": metrics["selected_model"], "test_rows": len(test),
                      "test_positives": int(test.target.sum()), "holdout": results}, indent=2), flush=True)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/american_bankruptcy.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    run_experiment(args.data, args.output, args.trials)
