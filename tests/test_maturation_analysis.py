from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from python.maturation_analysis import (
    DEFAULT_BMI_REFERENCE,
    DEFAULT_HEIGHT_REFERENCE,
    DEFAULT_PLAYERS,
    add_age_variables,
    add_bmi,
    build_processed_dataset,
    classify_growth_stage_proxy,
    evaluate_with_loocv,
    lms_zscore,
    load_table,
)


def test_growth_proxy_boundaries():
    assert classify_growth_stage_proxy(-1.0) == "Pre-PHV proxy"
    assert classify_growth_stage_proxy(-0.99) == "PHV proxy"
    assert classify_growth_stage_proxy(0.99) == "PHV proxy"
    assert classify_growth_stage_proxy(1.0) == "Post-PHV proxy"
    assert classify_growth_stage_proxy(np.nan) is None


def test_bmi_calculation():
    df = pd.DataFrame({"height_cm": [170.0], "weight_kg": [68.0]})
    out = add_bmi(df)
    assert out.loc[0, "bmi"] == 23.53


def test_age_variables():
    df = pd.DataFrame({"date_of_birth": ["2011-04-12"]})
    out = add_age_variables(df, datetime(2025, 4, 12))
    assert out.loc[0, "age_decimal"] == 14.0
    assert out.loc[0, "age_months"] == 168


def test_lms_zscore_l_zero():
    z = lms_zscore(value=20.0, l_value=0.0, m_value=18.0, s_value=0.1)
    assert np.isfinite(z)


def test_bundled_demo_runs_end_to_end():
    players = load_table(DEFAULT_PLAYERS)
    height_ref = load_table(DEFAULT_HEIGHT_REFERENCE)
    bmi_ref = load_table(DEFAULT_BMI_REFERENCE)
    processed = build_processed_dataset(players, height_ref, bmi_ref, datetime(2025, 4, 12))
    assert len(processed) == 26
    assert processed["growth_stage_proxy"].notna().all()
    report, confusion, classes, importance = evaluate_with_loocv(processed)
    assert confusion.shape == (len(classes), len(classes))
    assert "accuracy" in report
    assert not importance.empty
