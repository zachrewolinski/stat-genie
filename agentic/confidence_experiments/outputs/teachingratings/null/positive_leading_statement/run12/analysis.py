import json
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


df = pd.read_csv("teachingratings.csv")

# Basic stats
n = len(df)
mean_eval = df["eval"].mean()
std_eval = df["eval"].std(ddof=1)
std_beauty = df["beauty"].std(ddof=1)

# Pearson correlation
corr, corr_p = stats.pearsonr(df["beauty"], df["eval"])

# Simple OLS
model1 = smf.ols("eval ~ beauty", data=df).fit()

# OLS with controls
formula_controls = (
    "eval ~ beauty + age + C(gender) + C(minority) + C(native) + "
    "C(tenure) + C(division) + C(credits) + students + allstudents"
)
model2 = smf.ols(formula_controls, data=df).fit()
model2_robust = model2.get_robustcov_results(cov_type="HC3")

# Identify coefficient index for beauty in robust results
beauty_idx = model2_robust.model.exog_names.index("beauty")

# Collect results
results = {
    "n": n,
    "mean_eval": mean_eval,
    "std_eval": std_eval,
    "std_beauty": std_beauty,
    "corr": corr,
    "corr_p": corr_p,
    "model1_coef": model1.params["beauty"],
    "model1_p": model1.pvalues["beauty"],
    "model1_r2": model1.rsquared,
    "model2_coef": model2.params["beauty"],
    "model2_p": model2.pvalues["beauty"],
    "model2_r2": model2.rsquared,
    "model2_robust_coef": model2_robust.params[beauty_idx],
    "model2_robust_p": model2_robust.pvalues[beauty_idx],
    "model2_robust_r2": model2.rsquared,
}

# Effect size for 1 SD increase in beauty
results["effect_1sd"] = results["model2_coef"] * std_beauty
results["effect_1sd_robust"] = results["model2_robust_coef"] * std_beauty

print(json.dumps(results, indent=2))
