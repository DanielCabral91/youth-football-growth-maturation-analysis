"""Youth Football Growth & Maturation Analysis.

Portfolio refactor of an exploratory internship project.

Important
---------
The default run uses fully synthetic player data and synthetic growth-reference
files bundled with this repository so that the project is reproducible without
publishing any real youth-athlete data or redistributing third-party reference
workbooks.

The ``growth_stage_proxy`` reproduces the exploratory rule used in the
internship project: height z-score <= -1 -> Pre-PHV proxy, between -1 and 1 ->
PHV proxy, >= 1 -> Post-PHV proxy.

This is NOT a clinically validated biological-maturation diagnosis. The ML
section evaluates how well anthropometric variables reproduce those proxy
labels; it does not validate the proxy against a clinical gold standard.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAYERS = REPO_ROOT / "data" / "synthetic_players.csv"
DEFAULT_HEIGHT_REFERENCE = REPO_ROOT / "data" / "reference" / "synthetic_hfa_boys_reference.csv"
DEFAULT_BMI_REFERENCE = REPO_ROOT / "data" / "reference" / "synthetic_bmi_boys_reference.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "maturation"

COLUMN_ALIASES = {
    "Codinome": "player_id",
    "Data de Nascimento": "date_of_birth",
    "Altura (cm)": "height_cm",
    "Altura Sentada (cm)": "sitting_height_cm",
    "Peso (kg)": "weight_kg",
}

ML_FEATURES = [
    "age_decimal",
    "height_cm",
    "sitting_height_cm",
    "weight_kg",
    "bmi",
    "bmi_zscore",
]


def load_table(path: str | Path, sheet_name: int | str = 0) -> pd.DataFrame:
    """Load a CSV or Excel table."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    raise ValueError(f"Unsupported file format: {suffix}")


def normalize_player_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Accept original Portuguese column names or portfolio English names."""
    out = df.rename(columns=COLUMN_ALIASES).copy()
    required = {"player_id", "date_of_birth", "height_cm", "weight_kg"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"Missing required player columns: {sorted(missing)}")
    return out


def add_age_variables(df: pd.DataFrame, evaluation_date: datetime) -> pd.DataFrame:
    out = df.copy()
    out["date_of_birth"] = pd.to_datetime(out["date_of_birth"], errors="coerce")
    out["age_decimal"] = (evaluation_date - out["date_of_birth"]).dt.days / 365.25
    out["age_decimal"] = out["age_decimal"].round(2)
    out["age_months"] = (out["age_decimal"] * 12).round().astype("Int64")
    return out


def add_bmi(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    height_m = out["height_cm"] / 100.0
    out["bmi"] = (out["weight_kg"] / (height_m**2)).round(2)
    return out


def prepare_height_reference(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a height-for-age table with Month, P3, P50 and P97 columns.

    The SD approximation ``(P97 - P3) / 4`` is preserved from the internship
    workflow for methodological traceability. In the repository's default demo,
    these reference values are synthetic and exist only for reproducibility.
    """
    ref = df.copy()
    ref.columns = ref.columns.str.strip()
    required = {"Month", "P3", "P50", "P97"}
    missing = required.difference(ref.columns)
    if missing:
        raise ValueError(f"Missing height-reference columns: {sorted(missing)}")
    ref = ref[["Month", "P3", "P50", "P97"]].rename(
        columns={"Month": "age_months", "P50": "height_median"}
    )
    ref["age_months"] = ref["age_months"].astype(int)
    ref["height_sd_estimated"] = (ref["P97"] - ref["P3"]) / 4.0
    return ref


def add_height_zscore(df: pd.DataFrame, height_reference: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(height_reference, how="left", on="age_months", validate="many_to_one")
    out["height_zscore"] = (
        (out["height_cm"] - out["height_median"]) / out["height_sd_estimated"]
    ).round(2)
    return out


def prepare_bmi_lms_reference(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare an LMS BMI-for-age reference table with Month, L, M, S columns."""
    ref = df.copy()
    ref.columns = ref.columns.str.strip()
    required = {"Month", "L", "M", "S"}
    missing = required.difference(ref.columns)
    if missing:
        raise ValueError(f"Missing BMI-reference columns: {sorted(missing)}")
    ref = ref[["Month", "L", "M", "S"]].rename(columns={"Month": "age_months"})
    ref["age_months"] = ref["age_months"].astype(int)
    return ref


def lms_zscore(value: float, l_value: float, m_value: float, s_value: float) -> float:
    """Compute an LMS z-score."""
    if any(pd.isna(v) for v in [value, l_value, m_value, s_value]):
        return np.nan
    if m_value <= 0 or s_value <= 0 or value <= 0:
        return np.nan
    if l_value != 0:
        return (((value / m_value) ** l_value) - 1.0) / (l_value * s_value)
    return np.log(value / m_value) / s_value


def add_bmi_zscore(df: pd.DataFrame, bmi_reference: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(bmi_reference, how="left", on="age_months", validate="many_to_one")
    out["bmi_zscore"] = out.apply(
        lambda row: lms_zscore(row["bmi"], row["L"], row["M"], row["S"]), axis=1
    )
    out["bmi_zscore"] = out["bmi_zscore"].round(2)
    return out


def classify_growth_stage_proxy(height_zscore: float) -> str | None:
    """Reproduce the exploratory threshold rule used in the internship project."""
    if pd.isna(height_zscore):
        return None
    if height_zscore <= -1:
        return "Pre-PHV proxy"
    if height_zscore >= 1:
        return "Post-PHV proxy"
    return "PHV proxy"


def build_processed_dataset(
    players: pd.DataFrame,
    height_reference: pd.DataFrame,
    bmi_reference: pd.DataFrame,
    evaluation_date: datetime,
) -> pd.DataFrame:
    out = normalize_player_columns(players)
    out = add_age_variables(out, evaluation_date)
    out = add_bmi(out)
    out = add_height_zscore(out, prepare_height_reference(height_reference))
    out = add_bmi_zscore(out, prepare_bmi_lms_reference(bmi_reference))
    out["growth_stage_proxy"] = out["height_zscore"].apply(classify_growth_stage_proxy)
    return out


def make_classifier(num_classes: int) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=1,
    )


def evaluate_with_loocv(df: pd.DataFrame) -> tuple[dict, np.ndarray, list[str], pd.Series]:
    """Evaluate XGBoost with LOOCV without direct target leakage.

    ``height_zscore`` is intentionally excluded from ``ML_FEATURES`` because it
    directly defines the target proxy. The remaining features still contain
    information related to the rule, so the exercise should be understood as
    reproduction of a derived proxy label, not validation of maturity status.
    """
    model_data = df[ML_FEATURES + ["growth_stage_proxy"]].dropna().copy()
    if len(model_data) < 3:
        raise ValueError("Not enough complete rows for LOOCV.")

    X = model_data[ML_FEATURES]
    y = model_data["growth_stage_proxy"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    if len(encoder.classes_) < 2:
        raise ValueError("LOOCV requires at least two target classes.")

    loo = LeaveOneOut()
    y_true: list[int] = []
    y_pred: list[int] = []

    for train_idx, test_idx in loo.split(X):
        model = make_classifier(len(encoder.classes_))
        model.fit(X.iloc[train_idx], y_encoded[train_idx])
        pred = model.predict(X.iloc[test_idx])
        y_true.extend(y_encoded[test_idx].tolist())
        y_pred.extend(np.asarray(pred, dtype=int).tolist())

    labels = list(range(len(encoder.classes_)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    final_model = make_classifier(len(encoder.classes_))
    final_model.fit(X, y_encoded)
    importance = pd.Series(final_model.feature_importances_, index=ML_FEATURES).sort_values()

    return report, cm, encoder.classes_.tolist(), importance


def save_outputs(
    processed: pd.DataFrame,
    report: dict,
    confusion: np.ndarray,
    class_names: list[str],
    importance: pd.Series,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed.to_csv(output_dir / "processed_players.csv", index=False)
    with open(output_dir / "loocv_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion)
    ax.set_xticks(range(len(class_names)), labels=class_names, rotation=25, ha="right")
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed proxy label")
    ax.set_title("LOOCV confusion matrix")
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(j, i, int(confusion[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "loocv_confusion_matrix.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    importance.plot(kind="barh", ax=ax)
    ax.set_xlabel("XGBoost feature importance")
    ax.set_title("Descriptive feature importance — final fit")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=160)
    plt.close(fig)

    counts = processed["growth_stage_proxy"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Exploratory growth-stage proxy")
    ax.set_ylabel("Players")
    ax.set_title("Proxy-label distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "proxy_distribution.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--players",
        default=str(DEFAULT_PLAYERS),
        help="CSV/XLSX player dataset (default: bundled synthetic demo data)",
    )
    parser.add_argument(
        "--height-reference",
        default=str(DEFAULT_HEIGHT_REFERENCE),
        help="Height-for-age reference table (default: bundled synthetic demo reference)",
    )
    parser.add_argument(
        "--bmi-reference",
        default=str(DEFAULT_BMI_REFERENCE),
        help="BMI-for-age LMS table (default: bundled synthetic demo reference)",
    )
    parser.add_argument("--evaluation-date", default="2025-04-12", help="YYYY-MM-DD")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluation_date = datetime.strptime(args.evaluation_date, "%Y-%m-%d")

    players = load_table(args.players)
    height_reference = load_table(args.height_reference)
    bmi_reference = load_table(args.bmi_reference)

    processed = build_processed_dataset(
        players=players,
        height_reference=height_reference,
        bmi_reference=bmi_reference,
        evaluation_date=evaluation_date,
    )
    report, confusion, class_names, importance = evaluate_with_loocv(processed)
    save_outputs(processed, report, confusion, class_names, importance, args.output_dir)

    print(f"Processed {len(processed)} synthetic demo players.")
    print(f"Outputs saved to: {args.output_dir}")
    print("Reminder: growth_stage_proxy is exploratory and not a clinical diagnosis.")


if __name__ == "__main__":
    main()
