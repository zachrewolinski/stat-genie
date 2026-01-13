from typing import Any, Dict, List, Optional
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to create analysis-ready columns.

    Required FINAL columns produced by this function (must not be renamed):
      - 'Enrollment'
      - 'NumTeachers'
      - 'PctCalWorks'
      - 'PctReducedLunch'
      - 'NumComputers'
      - 'ExpenditurePerStudent'
      - 'DistrictIncomeK'
      - 'PctEnglishLearners'
      - 'ReadingScore'
      - 'MathScore'
      - 'County'
      - 'GradeSpan'
      - 'StudentTeacherRatio'
      - 'AvgTestScore'
      - 'ComputersPerStudent'
      - 'LogEnrollment'
      - 'LogExpenditurePerStudent'
      - 'LogComputersPerStudent'

    This function is robust to common variations in raw column names by
    attempting to match plausible alternatives for each required input.
    """
    df = df.copy()

    # Helper to normalize names for matching
    def _norm(s: str) -> str:
        return re.sub(r"\W+", "", str(s)).lower()

    # Build a lookup of normalized existing column names to actual names
    existing_cols = {_norm(c): c for c in df.columns}

    # Potential source names for each target final column.
    # These are heuristics to handle datasets that don't use 'featureX' names.
    candidates: Dict[str, List[str]] = {
        "Enrollment": [
            "feature6",
            "enrollment",
            "enroll",
            "totalstudents",
            "total_students",
            "students",
            "studentcount",
        ],
        "NumTeachers": [
            "feature7",
            "numteachers",
            "num_teachers",
            "teachers",
            "fte_teachers",
            "teachers_fte",
            "teacher",
        ],
        "PctCalWorks": [
            "feature8",
            "pctcalworks",
            "pct_calworks",
            "calworks",
            "cal_works",
        ],
        "PctReducedLunch": [
            "feature9",
            "pctreducedlunch",
            "pct_reduced_lunch",
            "reducedlunch",
            "reduced_lunch",
            "lunch",
        ],
        "NumComputers": [
            "feature10",
            "numcomputers",
            "num_computers",
            "computers",
            "computer",
            "num_computer",
        ],
        "ExpenditurePerStudent": [
            "feature11",
            "expenditureperstudent",
            "expenditure_per_student",
            "expenditure",
            "exp_per_student",
            "spend_per_student",
        ],
        "DistrictIncomeK": [
            "feature12",
            "districtincomek",
            "district_income_k",
            "district_income",
            "income_k",
            "income",
            "avg_income",
        ],
        "PctEnglishLearners": [
            "feature13",
            "pctenglishlearners",
            "pct_english_learners",
            "englishlearners",
            "english_learners",
            "english",
        ],
        "ReadingScore": [
            "feature14",
            "readingscore",
            "reading_score",
            "avg_reading",
            "reading",
            "read",
        ],
        "MathScore": [
            "feature15",
            "mathscore",
            "math_score",
            "avg_math",
            "math",
        ],
        "County": ["feature4", "county", "region"],
        "GradeSpan": [
            "feature5",
            "gradespan",
            "grade_span",
            "grade",
            "grades",
            "gradelevel",
        ],
    }

    found_map: Dict[str, Optional[str]] = {}
    for target, prefs in candidates.items():
        found = None
        for p in prefs:
            p_norm = _norm(p)
            if p_norm in existing_cols:
                found = existing_cols[p_norm]
                break
        # Also try exact match of target name as a last resort
        if found is None:
            for col in df.columns:
                if col == target:
                    found = col
                    break
        found_map[target] = found

    missing_raw = [t for t, v in found_map.items() if v is None]
    if missing_raw:
        raise ValueError(
            "Could not find raw columns required to construct the final dataframe. "
            f"Missing mappings for: {missing_raw}. Available columns: {list(df.columns)}"
        )

    # Copy and coerce numeric columns to numeric dtype (errors -> NaN)
    numeric_targets = [
        "Enrollment",
        "NumTeachers",
        "PctCalWorks",
        "PctReducedLunch",
        "NumComputers",
        "ExpenditurePerStudent",
        "DistrictIncomeK",
        "PctEnglishLearners",
        "ReadingScore",
        "MathScore",
    ]
    for t in numeric_targets:
        raw_col = found_map[t]
        df[t] = pd.to_numeric(df[raw_col], errors="coerce")

    # For categorical columns, copy raw values
    df["County"] = df[found_map["County"]].astype("category")
    df["GradeSpan"] = df[found_map["GradeSpan"]].astype("category")

    # Drop rows missing the essential variables used to compute IV and DV
    df = df.dropna(subset=["Enrollment", "NumTeachers", "ReadingScore", "MathScore"])

    # Avoid division by zero and create StudentTeacherRatio (students per teacher)
    # Treat nonpositive teacher counts as missing
    df.loc[df["NumTeachers"] <= 0, "NumTeachers"] = np.nan
    df["StudentTeacherRatio"] = df["Enrollment"] / df["NumTeachers"]

    # Create dependent variable: average of reading and math scores
    df["AvgTestScore"] = df[["ReadingScore", "MathScore"]].mean(axis=1)

    # Create computers per student (handle division by zero)
    df.loc[df["Enrollment"] <= 0, "Enrollment"] = np.nan
    df["ComputersPerStudent"] = df["NumComputers"] / df["Enrollment"]

    # Log transformations for skewed controls (use log1p to handle zeros)
    df["LogEnrollment"] = np.log1p(df["Enrollment"])
    df["LogExpenditurePerStudent"] = np.log1p(df["ExpenditurePerStudent"].astype(float))
    # Fillna(0) so log1p works; if ComputersPerStudent is NaN it stays NaN after fill
    df["LogComputersPerStudent"] = np.log1p(df["ComputersPerStudent"].fillna(0))

    # Keep only rows without missing values in model columns
    model_cols = [
        "AvgTestScore",
        "StudentTeacherRatio",
        "PctCalWorks",
        "PctReducedLunch",
        "ComputersPerStudent",
        "ExpenditurePerStudent",
        "DistrictIncomeK",
        "PctEnglishLearners",
        "Enrollment",
        "LogEnrollment",
        "County",
        "GradeSpan",
    ]
    df = df.dropna(subset=model_cols)

    # Ensure categorical columns are typed properly (again)
    df["County"] = df["County"].astype("category")
    df["GradeSpan"] = df["GradeSpan"].astype("category")

    # Return final dataframe with required columns (preserve any extra columns)
    return df


def model(df: pd.DataFrame) -> Any:
    """
    Runs an OLS regression of AvgTestScore on StudentTeacherRatio controlling for
    socioeconomic and resource variables and including county and grade-span fixed effects.

    Returns the fitted regression results object with heteroskedasticity-robust SEs (HC3).
    """
    required = [
        "AvgTestScore",
        "StudentTeacherRatio",
        "PctCalWorks",
        "PctReducedLunch",
        "ComputersPerStudent",
        "ExpenditurePerStudent",
        "DistrictIncomeK",
        "PctEnglishLearners",
        "LogEnrollment",
        "County",
        "GradeSpan",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    formula = (
        "AvgTestScore ~ StudentTeacherRatio + PctCalWorks + PctReducedLunch + "
        "ComputersPerStudent + ExpenditurePerStudent + DistrictIncomeK + "
        "PctEnglishLearners + LogEnrollment + C(County) + C(GradeSpan)"
    )

    # Fit OLS and then obtain HC3 robust covariance results
    ols_res = smf.ols(formula=formula, data=df).fit()
    robust_res = ols_res.get_robustcov_results(cov_type="HC3")

    return robust_res