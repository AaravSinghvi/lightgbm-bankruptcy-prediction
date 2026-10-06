# Data source and target definition

The input file is `american_bankruptcy.csv`, obtained from Kaggle according to the project author:

- [Author-supplied Kaggle source](https://www.kaggle.com/code/utkarshx27/us-company-bankruptcy-prediction-93-accuracy/input)
- [Associated dataset listing](https://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset)

Download the CSV from Kaggle and put it at `data/american_bankruptcy.csv`. The raw data is not redistributed in this repository. Follow the source's terms when obtaining and using it.

The tested local copy contains 78,682 rows, 8,971 distinct anonymized companies, and 20 calendar years from 1999 through 2018. It has `company_name`, `status_label`, `year`, and financial columns `X1` through `X18`. A different version of this dataset may use different column ordering, units, and names; do not substitute it without checking the schema and definitions.

SHA-256 of the tested CSV:

`cff2c899a97ecd629415cb22f59186000e74e1c0a78cfae036c0a53025419b5e`

## Cohort

The original notebook interprets `X8` as market value and uses a company median between 250 and 2,000 as a small-cap proxy. The revised workflow retains those bounds, but computes the median using only records available through each training cutoff. It then evaluates later records of the selected companies. The numeric units and market-cap interpretation should be checked against the exact Kaggle version; no external market-cap definition is asserted here.

## Target

In the provided CSV, status is constant within each company. The original project changes that into a row target: only the company's final observed row is positive if its status is `failed`; every other row is negative. The revised workflow preserves and explicitly names this assumption.

This is a **retrospective terminal-record proxy**. Exact filing dates, feature reporting dates, and label-availability timestamps are not supplied. Relabeling does not independently establish a one-year-ahead forecasting target, and records near a cutoff may have labels that would not have been known at that time. Correct chronological folds do not resolve that limitation.

Companies can appear in both training and later evaluation years. The experiment measures later records of known companies rather than generalization to entirely unseen companies. Positive examples are sparse, so per-company dependence and sampling uncertainty matter.
