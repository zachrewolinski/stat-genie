import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = "teachingratings.csv"

df = pd.read_csv(DATA_PATH)

# Basic cleaning/transformations
# Ensure categorical variables are treated as categories
cat_cols = ["minority", "gender", "credits", "division", "native", "tenure"]
for c in cat_cols:
    df[c] = df[c].astype("category")

# Log transforms for size measures to reduce skew
# Add 1 to avoid log(0) though min is >0
for c in ["students", "allstudents"]:
    df[f"log_{c}"] = np.log(df[c])

# Simple correlation
corr, corr_p = stats.pearsonr(df["beauty"], df["eval"])

# Simple OLS
model_simple = smf.ols("eval ~ beauty", data=df).fit(cov_type="HC3")

# Multiple OLS with controls
formula = (
    "eval ~ beauty + age + C(gender) + C(minority) + C(native) + "
    "C(tenure) + C(division) + C(credits) + log_students + log_allstudents"
)
model_controls = smf.ols(formula, data=df).fit(cov_type="HC3")

# Clustered SEs by professor as a robustness check
try:
    model_cluster = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["prof"]}
    )
except Exception:
    model_cluster = None

# Standardized effect: beauty is already centered; compute SDs
beauty_sd = df["beauty"].std(ddof=1)
eval_sd = df["eval"].std(ddof=1)

# Effect size per 1 SD of beauty in controls model
coef_controls = model_controls.params.get("beauty", np.nan)
se_controls = model_controls.bse.get("beauty", np.nan)
p_controls = model_controls.pvalues.get("beauty", np.nan)

# Convert to eval SD units
beta_std = coef_controls * beauty_sd / eval_sd

results = {
    "n": int(df.shape[0]),
    "corr": float(corr),
    "corr_p": float(corr_p),
    "simple_coef": float(model_simple.params["beauty"]),
    "simple_se": float(model_simple.bse["beauty"]),
    "simple_p": float(model_simple.pvalues["beauty"]),
    "controls_coef": float(coef_controls),
    "controls_se": float(se_controls),
    "controls_p": float(p_controls),
    "beta_std": float(beta_std),
}

if model_cluster is not None:
    results.update(
        {
            "cluster_coef": float(model_cluster.params.get("beauty", np.nan)),
            "cluster_se": float(model_cluster.bse.get("beauty", np.nan)),
            "cluster_p": float(model_cluster.pvalues.get("beauty", np.nan)),
        }
    )

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
