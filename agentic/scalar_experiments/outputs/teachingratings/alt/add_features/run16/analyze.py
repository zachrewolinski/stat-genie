import json
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

DATA_PATH = Path(__file__).with_name("teachingratings.csv")

# Load data
_df = pd.read_csv(DATA_PATH)

# Core variables
# Ensure expected columns exist
required_cols = ["beauty", "eval"]
missing = [c for c in required_cols if c not in _df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Drop rows with missing key variables
_df = _df.dropna(subset=required_cols).copy()

# Simple correlation
corr = _df["beauty"].corr(_df["eval"])

# Simple OLS: eval ~ beauty
model_simple = smf.ols("eval ~ beauty", data=_df).fit()

# Build a more controlled model using typical covariates from dataset
# Include demographic/instructor and course characteristics when available.
# C() for categorical variables.
formula_terms = [
    "beauty",
    "age",
    "C(gender)",
    "C(minority)",
    "C(native)",
    "C(tenure)",
    "C(division)",
    "C(credits)",
    "students",
    "allstudents",
]

# Keep only columns that exist in the dataset
usable_terms = []
for term in formula_terms:
    # if term is like C(col)
    if term.startswith("C(") and term.endswith(")"):
        col = term[2:-1]
        if col in _df.columns:
            usable_terms.append(term)
    else:
        if term in _df.columns:
            usable_terms.append(term)

formula = "eval ~ " + " + ".join(usable_terms)
model_controls = smf.ols(formula, data=_df).fit()

# Collect key results
results = {
    "n": int(model_simple.nobs),
    "corr_beauty_eval": float(corr),
    "simple": {
        "coef_beauty": float(model_simple.params.get("beauty", float("nan"))),
        "p_beauty": float(model_simple.pvalues.get("beauty", float("nan"))),
        "r2": float(model_simple.rsquared),
    },
    "controls": {
        "formula": formula,
        "coef_beauty": float(model_controls.params.get("beauty", float("nan"))),
        "p_beauty": float(model_controls.pvalues.get("beauty", float("nan"))),
        "r2": float(model_controls.rsquared),
    },
}

print(json.dumps(results, indent=2))
