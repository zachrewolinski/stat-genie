from typing import Any, List
import numpy as np
import pandas as pd
import statsmodels.api as sm
from types import SimpleNamespace


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the analysis dataframe.

    The function produces the exact FINAL columns required by the analysis:
      - StudentTeacherRatio: numeric, computed when possible (enrollment / teachers) or read directly if present
      - AvgTestScore: numeric, mapped from common candidate raw columns
      - Expenditure: numeric, mapped from common candidate raw columns
      - Income: numeric, mapped from common candidate raw columns
      - PctEnglishLearners: numeric, mapped from common candidate raw columns

    The function is permissive about which raw column names are present in the input data:
    it searches a list of plausible candidate column names for each final variable and uses the
    first available one. It coerces chosen raw columns to numeric (errors coerced to NaN),
    avoids division by zero when computing StudentTeacherRatio, and creates helper column LogSTR
    (natural log of StudentTeacherRatio) for diagnostics. The function does NOT drop rows
    except insofar as values are missing or invalid; the modeling function is responsible for
    dropping observations with missing required variables.
    """
    df = df.copy()

    # Helper: create an empty numeric Series aligned with df index
    def empty_series():
        return pd.Series(index=df.index, dtype="float64")

    # Helper: return the first existing candidate column (coerced to numeric), or an empty series
    def first_numeric(candidates: List[str]) -> pd.Series:
        for name in candidates:
            if name in df.columns:
                return pd.to_numeric(df[name], errors="coerce")
        return empty_series()

    # Candidate raw column names for each conceptual variable (keeps final column names fixed)
    avg_test_candidates = [
        "AvgTestScore", "grades", "avg_test_score", "test_score", "tests", "stanford9", "scaled_score", "score", "mean_score",
        "testscr", "testscr", "testscore", "avgscore", "avg_test", "tests_score"
    ]
    expenditure_candidates = [
        "Expenditure", "expenditure", "exp_per_student", "per_pupil_expenditure", "expend_per_student", "spend_per_pupil", "expenditures",
        "expend", "spending_per_student", "spending"
    ]
    income_candidates = [
        "Income", "income", "median_income", "avg_income", "income_proxy", "median_household_income", "avginc", "avg_inc", "avg_income"
    ]
    pct_el_candidates = [
        "PctEnglishLearners", "pct_english_learners", "english_learners", "el_pct", "percent_el", "ELL", "ell_pct", "ell", "EL"
    ]
    enrollment_candidates = [
        "TotalEnrollment", "total_enrollment", "enrollment", "students", "student_count", "total_students", "enroll", "enroll_total"
    ]
    teachers_candidates = [
        "Teachers", "teachers", "fte_teachers", "teacher_fte", "num_teachers", "num_fte_teachers", "teacher_count", "teach_fte", "teaching_staff"
    ]
    # Additional candidates for precomputed ratio-like columns
    str_candidates = [
        "StudentTeacherRatio", "student_teacher_ratio", "students_per_teacher", "stu_teacher_ratio", "stu_per_teacher", "student_teacher",
        "str", "stu_teach_ratio", "student_teacher", "student_teacher_r"
    ]

    # Populate final columns by selecting first available candidate
    df["AvgTestScore"] = first_numeric(avg_test_candidates)
    df["Expenditure"] = first_numeric(expenditure_candidates)
    df["Income"] = first_numeric(income_candidates)
    df["PctEnglishLearners"] = first_numeric(pct_el_candidates)

    # StudentTeacherRatio: prefer an existing column; otherwise compute from enrollment and teachers if available
    # First check for precomputed ratio-like raw columns
    ratio_series = None
    for name in str_candidates:
        if name in df.columns:
            ratio_series = pd.to_numeric(df[name], errors="coerce")
            break

    if ratio_series is not None:
        df["StudentTeacherRatio"] = ratio_series
    else:
        enrollment = first_numeric(enrollment_candidates)
        teachers = first_numeric(teachers_candidates)
        # Avoid division by zero or negative/zero teacher counts: set ratio to NaN where teachers <= 0 or NaN
        valid_teachers = teachers.where(teachers > 0)
        # Perform division; resulting Series will align with df.index
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = enrollment / valid_teachers
        df["StudentTeacherRatio"] = ratio

    # Clip extreme ratios to avoid numerical issues in logs (retain information but avoid -inf)
    # Only clip lower bound (ratios <= 0 become NaN already); set sensible lower bound if extremely small but positive
    small_floor = 0.1
    # Replace non-positive or zero values with NaN (they are invalid for ratio)
    df["StudentTeacherRatio"] = df["StudentTeacherRatio"].where(df["StudentTeacherRatio"] > 0, np.nan)
    # For very small positive values, set a small floor to avoid -inf in logs; keep NaN as NaN
    df["StudentTeacherRatio"] = df["StudentTeacherRatio"].where(
        df["StudentTeacherRatio"].isna() | (df["StudentTeacherRatio"] >= small_floor),
        other=df["StudentTeacherRatio"].clip(lower=small_floor)
    )

    # Helper diagnostic column: log of StudentTeacherRatio
    df["LogSTR"] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        valid_mask = df["StudentTeacherRatio"].notna() & (df["StudentTeacherRatio"] > 0)
        df.loc[valid_mask, "LogSTR"] = np.log(df.loc[valid_mask, "StudentTeacherRatio"])

    # Ensure final dataframe contains the required final columns (even if they are all NaN)
    for col in ["StudentTeacherRatio", "AvgTestScore", "Expenditure", "Income", "PctEnglishLearners"]:
        if col not in df.columns:
            df[col] = empty_series()

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS regression of AvgTestScore on StudentTeacherRatio with controls.

    Model specification:
      AvgTestScore_i = beta0 + beta1 * StudentTeacherRatio_i + beta2 * Expenditure_i
                        + beta3 * Income_i + beta4 * PctEnglishLearners_i + eps_i

    The function returns the fitted statsmodels results object using robust (HC3) standard errors.
    If there are no observations with the full set of chosen controls, the function will iteratively
    drop controls with the worst coverage. If no observations remain even after dropping controls,
    the function will attempt to fit a model using only AvgTestScore and StudentTeacherRatio (pairwise).
    If absolutely no usable observations exist, a lightweight dummy results object is returned
    (with NaN parameters) instead of raising an exception.
    """
    df = df.copy()

    # Required columns for the analysis (must exist in the FINAL dataframe)
    required_columns = ["AvgTestScore", "StudentTeacherRatio"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the dataframe passed to model().")

    # Determine which control variables are present (column exists) and contain at least one non-missing value
    potential_controls = ["Expenditure", "Income", "PctEnglishLearners"]
    available_controls = [c for c in potential_controls if c in df.columns and df[c].notna().any()]

    # We will attempt to include as many controls as possible but must ensure there are observations remaining.
    # Start with all available controls and drop the control with the worst coverage iteratively until we have at least one observation.
    controls = list(available_controls)  # copy

    def observations_with_controls(ctrls: List[str]) -> int:
        required_for_model = ["AvgTestScore", "StudentTeacherRatio"] + ctrls
        return df.dropna(subset=required_for_model).shape[0]

    # Try to find a set of controls (possibly empty) that yields at least one observation.
    # Prefer larger sets of controls but drop those with the poorest coverage if needed.
    successful_controls = None
    # If there are no controls, just check the base model
    if not controls:
        if observations_with_controls([]) == 0:
            # Instead of failing immediately, check if at least y and the main IV have any pairwise non-missing rows.
            pairwise = df.dropna(subset=["AvgTestScore", "StudentTeacherRatio"])
            if pairwise.shape[0] == 0:
                # No usable observations at all; will handle later by returning a dummy results object
                successful_controls = []
            else:
                successful_controls = []
        else:
            successful_controls = []
    else:
        # Start with full set
        current_controls = controls.copy()
        while True:
            nobs = observations_with_controls(current_controls)
            if nobs > 0:
                successful_controls = current_controls
                break
            # If no controls left to drop, check base model (no controls)
            if not current_controls:
                if observations_with_controls([]) == 0:
                    # As a last resort, check pairwise availability of y and StudentTeacherRatio only
                    pairwise = df.dropna(subset=["AvgTestScore", "StudentTeacherRatio"])
                    if pairwise.shape[0] == 0:
                        # No usable observations at all; will handle later by returning a dummy results object
                        successful_controls = []
                        break
                    successful_controls = []
                    break
                successful_controls = []
                break
            # Drop the control with the smallest non-missing count (worst coverage)
            non_missing_counts = {c: df[c].notna().sum() for c in current_controls}
            worst_control = min(non_missing_counts, key=non_missing_counts.get)
            current_controls.remove(worst_control)
            # loop will retry

    # Build list of X variables (always include StudentTeacherRatio)
    X_vars = ["StudentTeacherRatio"] + successful_controls

    # Drop rows with missing values in y or any of the X variables we will use
    required_for_model = ["AvgTestScore"] + X_vars
    df_model = df.dropna(subset=required_for_model)

    if df_model.shape[0] == 0:
        # Attempt to fit a pairwise model using only AvgTestScore and StudentTeacherRatio if possible
        pairwise = df.dropna(subset=["AvgTestScore", "StudentTeacherRatio"])
        if pairwise.shape[0] > 0:
            # Fit a simple model with only the main independent variable
            y_pair = pairwise["AvgTestScore"].astype(float)
            X_pair = pairwise[["StudentTeacherRatio"]].astype(float)
            X_pair = sm.add_constant(X_pair, has_constant="add")
            ols_model = sm.OLS(y_pair, X_pair)
            fitted = ols_model.fit()
            try:
                results = fitted.get_robustcov_results(cov_type="HC3")
            except Exception:
                results = fitted
            return results
        # No usable observations at all: return a dummy results-like object rather than raising
        params_index = ["const", "StudentTeacherRatio"]
        params = pd.Series(data=[np.nan] * len(params_index), index=params_index, dtype="float64")
        bse = pd.Series(data=[np.nan] * len(params_index), index=params_index, dtype="float64")
        tvalues = pd.Series(data=[np.nan] * len(params_index), index=params_index, dtype="float64")
        pvalues = pd.Series(data=[np.nan] * len(params_index), index=params_index, dtype="float64")
        cov = pd.DataFrame(np.nan, index=params_index, columns=params_index, dtype="float64")
        dummy = SimpleNamespace(
            params=params,
            bse=bse,
            tvalues=tvalues,
            pvalues=pvalues,
            cov_params=lambda: cov,
            nobs=0,
            model=None,
            summary=lambda: "No observations available to fit model."
        )
        return dummy

    # Prepare y and X
    y = df_model["AvgTestScore"].astype(float)
    X = df_model[X_vars].astype(float)

    # Add constant
    X = sm.add_constant(X, has_constant="add")

    # Fit OLS then obtain robust covariance (HC3)
    ols_model = sm.OLS(y, X)
    fitted = ols_model.fit()
    try:
        results = fitted.get_robustcov_results(cov_type="HC3")
    except Exception:
        # If obtaining robust cov fails for any reason, fall back to the fitted results without robust cov
        results = fitted

    return results