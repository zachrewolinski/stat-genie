import pandas as pd
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv("panda_nuts.csv")

# Compute nut-cracking efficiency: nuts opened per second
_df["efficiency"] = _df["feature5"] / _df["feature6"]

# Clean categorical variables
_df["sex"] = _df["feature3"].astype("category")
_df["help"] = _df["feature7"].astype("category")

# Fit linear model with age, sex, and help
model = smf.ols("efficiency ~ feature2 + C(sex) + C(help)", data=_df).fit()

print(model.summary())

# Save key results for manual inspection if needed
results = {
    "n": int(_df.shape[0]),
    "efficiency_mean": float(_df["efficiency"].mean()),
    "params": model.params.to_dict(),
    "pvalues": model.pvalues.to_dict(),
    "f_pvalue": float(model.f_pvalue),
    "r2": float(model.rsquared),
}

pd.Series(results).to_json("analysis_results.json")
