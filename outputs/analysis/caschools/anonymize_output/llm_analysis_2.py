from typing import Any, List, Optional
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the dataframe used for modeling.

    Produces the following REQUIRED final columns (must not be renamed):
      - StuTeacherRatio
      - AvgScore
      - PctReducedLunch
      - PctEnglishLearners
      - ExpPerStudent
      - AvgIncome
      - ComputersPerStudent

    This function is robust to a variety of input column names: it will attempt
    to locate the needed source columns (e.g., enrollment, teachers, reading,
    math, etc.) using a set of candidate keywords. If the required source
    variables cannot be found, it raises a clear error describing missing
    inputs.
    """
    df = df.copy()

    # Define the conceptual source variables we need and candidate keywords
    source_requirements = {
        "feature6": ["feature6", "enroll", "student", "students", "total_enrollment", "totalenrollment", "enrollment", "student_count"],
        "feature7": ["feature7", "teacher", "teachers", "num_teachers", "fte_teachers", "staff_teachers"],
        "feature14": ["feature14", "reading", "read", "reading_score", "avg_reading"],
        "feature15": ["feature15", "math", "mathematics", "math_score", "avg_math"],
        "feature9": ["feature9", "reduced", "reduced_lunch", "free_reduced", "pct_reduced", "percent_reduced", "free_lunch", "lunch"],
        "feature13": ["feature13", "english", "el", "english_learners", "pct_english", "percent_english"],
        "feature11": ["feature11", "expend", "expenditure", "exp_per", "exp_per_student", "spending_per_student", "spending"],
        "feature12": ["feature12", "income", "avg_income", "median_income", "household_income"],
        "feature10": ["feature10", "computer", "computers", "num_computers", "computers_total", "devices"]
    }

    def _normalize(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        return s.strip("_")

    normalized_cols = {col: _normalize(col) for col in df.columns}

    def find_column(candidates: List[str]) -> Optional[str]:
        # 1) exact normalized match
        cand_norms = [_normalize(c) for c in candidates]
        for col, ncol in normalized_cols.items():
            if ncol in cand_norms:
                return col
        # 2) substring match of keywords in normalized column names
        for col, ncol in normalized_cols.items():
            for keyword in cand_norms:
                if keyword and keyword in ncol:
                    return col
        # 3) try matching by digits if candidate contains only 'featureNN' format
        for col in df.columns:
            for cand in candidates:
                m1 = re.fullmatch(r"feature[_\- ]?(\d+)", cand, flags=re.IGNORECASE)
                m2 = re.fullmatch(r"feature[_\- ]?(\d+)", col, flags=re.IGNORECASE)
                if m1 and m2 and m1.group(1) == m2.group(1):
                    return col
        return None

    # Build mapping from expected source names to actual dataframe columns
    mapping = {}
    missing_expected = []
    for src, candidates in source_requirements.items():
        found = find_column(candidates)
        if found is None:
            missing_expected.append(src)
        else:
            mapping[src] = found

    if missing_expected:
        # If none of the expected source columns were found, provide detailed message listing attempted candidates.
        attempted = {k: source_requirements[k] for k in missing_expected}
        raise ValueError(f"Missing expected source columns in input dataframe for items: {missing_expected}. "
                         f"Attempted candidate names: {attempted}. Available columns: {list(df.columns)}")

    # Coerce the mapped source columns to numeric (bad values -> NaN)
    for src, col in mapping.items():
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Safe series references
    s6 = df[mapping["feature6"]]
    s7 = df[mapping["feature7"]]
    s10 = df[mapping["feature10"]]

    # Compute student-teacher ratio (students per teacher), guard against nonpositive teachers
    df["StuTeacherRatio"] = np.where(s7 > 0, s6 / s7, np.nan)

    # Compute AvgScore as mean of reading (feature14) and math (feature15)
    df["AvgScore"] = df[[mapping["feature14"], mapping["feature15"]]].mean(axis=1)

    # Controls: map source columns into the required final column names
    df["PctReducedLunch"] = df[mapping["feature9"]]
    df["PctEnglishLearners"] = df[mapping["feature13"]]
    df["ExpPerStudent"] = df[mapping["feature11"]]
    df["AvgIncome"] = df[mapping["feature12"]]

    # Computers per student (avoid division by zero)
    df["ComputersPerStudent"] = np.where(s6 > 0, s10 / s6, np.nan)

    # Keep only rows with the variables needed for modeling
    model_cols = [
        "AvgScore",
        "StuTeacherRatio",
        "PctReducedLunch",
        "PctEnglishLearners",
        "ExpPerStudent",
        "AvgIncome",
        "ComputersPerStudent",
    ]
    df = df.dropna(subset=model_cols)

    # Light winsorization to reduce influence of extreme outliers on ratios/resources
    def winsorize_series(s: pd.Series) -> pd.Series:
        if s.empty:
            return s
        lower = s.quantile(0.01)
        upper = s.quantile(0.99)
        return s.clip(lower=lower, upper=upper)

    df["StuTeacherRatio"] = winsorize_series(df["StuTeacherRatio"])
    df["ComputersPerStudent"] = winsorize_series(df["ComputersPerStudent"])
    df["ExpPerStudent"] = winsorize_series(df["ExpPerStudent"])
    df["AvgIncome"] = winsorize_series(df["AvgIncome"])

    # Final returned dataframe contains all original columns plus the derived variables
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a linear regression to estimate the association between student-teacher ratio and average test scores,
    controlling for socioeconomic and resource variables. Returns the fitted statsmodels regression results object.

    Model specification:
      AvgScore ~ StuTeacherRatio + PctReducedLunch + PctEnglishLearners + ExpPerStudent + AvgIncome + ComputersPerStudent

    Heteroskedasticity-robust standard errors (HC3) are used.
    """
    required = [
        "AvgScore",
        "StuTeacherRatio",
        "PctReducedLunch",
        "PctEnglishLearners",
        "ExpPerStudent",
        "AvgIncome",
        "ComputersPerStudent",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Transformed dataframe is missing required columns for modeling: {missing}")

    X = df[["StuTeacherRatio", "PctReducedLunch", "PctEnglishLearners", "ExpPerStudent", "AvgIncome", "ComputersPerStudent"]].copy()
    X = sm.add_constant(X)
    y = df["AvgScore"]

    ols_model = sm.OLS(y, X)
    results = ols_model.fit()
    robust_results = results.get_robustcov_results(cov_type="HC3")

    return robust_results