import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

DATA_PATH = "teachingratings.csv"

df = pd.read_csv(DATA_PATH)

# Basic checks
summary = {
    "n_rows": len(df),
    "n_cols": df.shape[1],
    "missing_by_col": df.isna().sum().to_dict(),
}

# Ensure columns are treated as expected
# Rename for readability in outputs
col_map = {
    "feature2": "minority",
    "feature3": "age",
    "feature4": "gender",
    "feature5": "single_credit",
    "feature6": "beauty",
    "feature7": "eval_score",
    "feature8": "division",
    "feature9": "native_english",
    "feature10": "tenure_track",
    "feature11": "n_rated",
    "feature12": "n_enrolled",
    "feature13": "instructor_id",
}

df = df.rename(columns=col_map)

# Correlation analyses
pearson_r, pearson_p = stats.pearsonr(df["beauty"], df["eval_score"])
spearman_rho, spearman_p = stats.spearmanr(df["beauty"], df["eval_score"])

# Simple OLS
model_simple = smf.ols("eval_score ~ beauty", data=df).fit()

# Multivariate OLS with controls
formula_controls = (
    "eval_score ~ beauty + age + C(gender) + C(minority) + C(single_credit) + "
    "C(division) + C(native_english) + C(tenure_track) + n_rated + n_enrolled"
)
model_controls = smf.ols(formula_controls, data=df).fit()

# Cluster-robust SEs by instructor (if repeated measures)
try:
    model_controls_cluster = smf.ols(formula_controls, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["instructor_id"]}
    )
except Exception:
    model_controls_cluster = None

results = {
    "summary": summary,
    "pearson": {"r": pearson_r, "p": pearson_p},
    "spearman": {"rho": spearman_rho, "p": spearman_p},
    "simple_ols": {
        "coef_beauty": model_simple.params.get("beauty"),
        "p_beauty": model_simple.pvalues.get("beauty"),
        "r2": model_simple.rsquared,
        "n": int(model_simple.nobs),
    },
    "controls_ols": {
        "coef_beauty": model_controls.params.get("beauty"),
        "p_beauty": model_controls.pvalues.get("beauty"),
        "r2": model_controls.rsquared,
        "n": int(model_controls.nobs),
    },
}

if model_controls_cluster is not None:
    results["controls_ols_cluster"] = {
        "coef_beauty": model_controls_cluster.params.get("beauty"),
        "p_beauty": model_controls_cluster.pvalues.get("beauty"),
        "r2": model_controls_cluster.rsquared,
        "n": int(model_controls_cluster.nobs),
    }

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
