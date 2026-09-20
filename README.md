# Project 01 — Youth Football Growth & Maturation Analysis

Exploratory football-data project reconstructed from work developed during a youth-football internship.

This repository demonstrates a complete public workflow for transforming anthropometric data into age-standardized indicators, generating an exploratory growth-stage proxy, and evaluating an XGBoost classifier with Leave-One-Out Cross-Validation (LOOCV).

> **Important:** this is a portfolio and methodological demonstration. The growth-stage labels are exploratory proxies and must not be interpreted as a clinically validated assessment of biological maturity.

## Project highlights

- End-to-end Python analysis pipeline
- 26 fully synthetic youth-player records for privacy-safe reproducibility
- Decimal age, BMI, height z-score and BMI LMS z-score calculations
- Exploratory three-class growth-stage proxy
- XGBoost multiclass classification
- Leave-One-Out Cross-Validation for a very small sample
- Automatic generation of tables and visual outputs
- Explicit treatment of target leakage and methodological limitations

## Technical workflow

`Player data → validation → decimal age → BMI → growth references → z-scores → exploratory proxy labels → XGBoost → LOOCV → outputs`

## Stack

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- matplotlib
- openpyxl
- pytest

## Reproducible public demo

The public repository does **not** contain the original internship database or identifiable information about youth players.

The default execution uses:

- `data/synthetic_players.csv` — 26 fictitious players
- `data/reference/synthetic_hfa_boys_reference.csv` — synthetic height-for-age demonstration values
- `data/reference/synthetic_bmi_boys_reference.csv` — synthetic LMS demonstration values

These reference files are **not official WHO or clinical reference data**. They exist only to make the public portfolio reproducible without redistributing third-party reference workbooks.

## Validation strategy

The original internship project involved a very small sample. A conventional train/test split would remove a substantial fraction of the available observations from model fitting.

For that reason, the public reconstruction uses **Leave-One-Out Cross-Validation (LOOCV)**: each observation is held out once for testing while the remaining observations are used for training.

LOOCV improves data efficiency for evaluation, but it does **not** eliminate the uncertainty associated with a small dataset.

### Synthetic-demo result

On the included synthetic dataset, the LOOCV pipeline produced:

| Metric | Result |
| --- | ---: |
| Accuracy | 61.5% |
| Macro precision | 62.5% |
| Macro recall | 63.9% |
| Macro F1-score | 62.3% |
| Sample size | 26 |

These values describe only the included **synthetic demonstration dataset**. They are not evidence of clinical validity or real-world predictive performance.

## Methodological safeguard: target leakage

The exploratory target is derived from `height_zscore`.

Therefore, `height_zscore` is **excluded from the XGBoost predictors**. Including it would allow the model to directly reconstruct the rule used to create the target and would lead to target leakage.

The machine-learning section therefore asks a narrower question:

> Can the remaining anthropometric variables reproduce the exploratory proxy labels under LOOCV?

It does **not** validate the proxy itself as a biological-maturity assessment.

## Example outputs

### Proxy-label distribution

![Proxy-label distribution](outputs/maturation/proxy_distribution.png)

### LOOCV confusion matrix

![LOOCV confusion matrix](outputs/maturation/loocv_confusion_matrix.png)

### XGBoost feature importance

![Feature importance](outputs/maturation/feature_importance.png)

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── python/
│   ├── __init__.py
│   └── maturation_analysis.py
├── data/
│   ├── README.md
│   ├── synthetic_players.csv
│   └── reference/
│       ├── synthetic_bmi_boys_reference.csv
│       └── synthetic_hfa_boys_reference.csv
├── docs/
│   └── maturation_methodology.md
├── tests/
│   ├── conftest.py
│   └── test_maturation_analysis.py
└── outputs/
    └── maturation/
        ├── processed_players.csv
        ├── loocv_classification_report.json
        ├── loocv_confusion_matrix.png
        ├── feature_importance.png
        └── proxy_distribution.png
```

## Run locally

Clone the repository, install the dependencies and run the pipeline:

```bash
pip install -r requirements.txt
python python/maturation_analysis.py
```

The script generates the processed dataset, LOOCV classification report, confusion matrix, proxy distribution and feature-importance plot in `outputs/maturation/`.

To inspect optional command-line arguments:

```bash
python python/maturation_analysis.py --help
```

## Tests

Run:

```bash
pytest
```

The tests cover key calculations and basic pipeline behaviour.

## Privacy

No identifiable youth-player data are published.

The names, dates of birth and anthropometric measurements in the public demonstration are synthetic and were created specifically for this repository.

## Methodology

A more detailed discussion of assumptions, limitations and the analytical design is available in:

[`docs/maturation_methodology.md`](docs/maturation_methodology.md)

## Project context

This repository is a portfolio reconstruction of an exploratory internship analysis. The public version prioritizes:

- reproducibility;
- privacy;
- methodological transparency;
- separation between exploratory analytics and validated assessment;
- clear communication of limitations.

The objective is to demonstrate the process of turning a practical sports-science question into a reproducible data-analysis workflow rather than to present a clinical maturity-diagnosis tool.
