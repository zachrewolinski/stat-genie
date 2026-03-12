import json
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

# Load data
csv_path = "teachingratings.csv"
df = pd.read_csv(csv_path)

# Basic cleaning
# Ensure categorical columns are treated as category/str
cat_cols = ["minority", "gender", "credits", "division", "native", "tenure"]
for c in cat_cols:
    df[c] = df[c].astype("category")

# Simple correlation
pearson_r, pearson_p = stats.pearsonr(df["beauty"], df["eval"])
spearman_r, spearman_p = stats.spearmanr(df["beauty"], df["eval"])

# Simple OLS
model_simple = smf.ols("eval ~ beauty", data=df)
res_simple = model_simple.fit(cov_type="cluster", cov_kwds={"groups": df["prof"]})

# Controlled OLS
formula_controls = (
    "eval ~ beauty + age + students + allstudents + "
    "C(minority) + C(gender) + C(credits) + C(division) + C(native) + C(tenure)"
)
model_ctrl = smf.ols(formula_controls, data=df)
res_ctrl = model_ctrl.fit(cov_type="cluster", cov_kwds={"groups": df["prof"]})

# Extract key stats
out = {
    "n": int(df.shape[0]),
    "beauty_mean": df["beauty"].mean(),
    "beauty_sd": df["beauty"].std(),
    "eval_mean": df["eval"].mean(),
    "eval_sd": df["eval"].std(),
    "pearson_r": pearson_r,
    "pearson_p": pearson_p,
    "spearman_r": spearman_r,
    "spearman_p": spearman_p,
    "simple_beta": res_simple.params["beauty"],
    "simple_p": res_simple.pvalues["beauty"],
    "simple_ci_low": res_simple.conf_int().loc["beauty", 0],
    "simple_ci_high": res_simple.conf_int().loc["beauty", 1],
    "simple_r2": res_simple.rsquared,
    "ctrl_beta": res_ctrl.params["beauty"],
    "ctrl_p": res_ctrl.pvalues["beauty"],
    "ctrl_ci_low": res_ctrl.conf_int().loc["beauty", 0],
    "ctrl_ci_high": res_ctrl.conf_int().loc["beauty", 1],
    "ctrl_r2": res_ctrl.rsquared,
}

# Effect per 1 SD of beauty on eval scale
out["simple_effect_1sd"] = out["simple_beta"] * out["beauty_sd"]
out["ctrl_effect_1sd"] = out["ctrl_beta"] * out["beauty_sd"]

print(json.dumps(out, indent=2))
