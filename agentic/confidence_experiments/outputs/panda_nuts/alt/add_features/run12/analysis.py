import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

DATA_PATH = "panda_nuts.csv"

df = pd.read_csv(DATA_PATH)

# Compute efficiency as nuts opened per second
# Avoid divide-by-zero; seconds min > 0 per metadata

df["efficiency"] = df["nuts_opened"] / df["seconds"]

# Ensure categorical variables
for col in ["sex", "help"]:
    df[col] = df[col].astype("category")

# Mixed effects model with random intercepts for chimpanzee
mixed_result = None
try:
    mixed_model = smf.mixedlm("efficiency ~ age + C(sex) + C(help)", df, groups=df["chimpanzee"])
    mixed_result = mixed_model.fit(reml=False, method="lbfgs")
except Exception as e:
    mixed_result = e

# OLS with cluster-robust SE by chimpanzee
ols_model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=df)
ols_result = ols_model.fit(cov_type="cluster", cov_kwds={"groups": df["chimpanzee"]})

# Joint F-test for all predictors (age, sex, help)
# Reference levels are captured in intercept; test the remaining coefficients.
# Build hypothesis matrix from parameter names excluding Intercept
param_names = [p for p in ols_result.params.index if p != "Intercept"]
# F-test for all non-intercept coefficients = 0
f_test = ols_result.f_test(" + ".join([f"{p} = 0" for p in param_names]))

# Summaries
output = {
    "n_rows": int(df.shape[0]),
    "n_chimps": int(df["chimpanzee"].nunique()),
    "efficiency_mean": float(df["efficiency"].mean()),
    "efficiency_std": float(df["efficiency"].std()),
    "ols_params": ols_result.params.to_dict(),
    "ols_pvalues": ols_result.pvalues.to_dict(),
    "ols_f_test": {
        "fvalue": float(np.asarray(f_test.fvalue)[0][0]) if hasattr(f_test.fvalue, "__array__") else float(f_test.fvalue),
        "pvalue": float(f_test.pvalue),
        "df_num": int(f_test.df_num),
        "df_denom": int(f_test.df_denom),
    },
}

if isinstance(mixed_result, Exception):
    output["mixed_error"] = str(mixed_result)
else:
    output["mixed_params"] = mixed_result.params.to_dict()
    output["mixed_pvalues"] = mixed_result.pvalues.to_dict()

with open("analysis_results.json", "w") as f:
    json.dump(output, f, indent=2)

print(json.dumps(output, indent=2))
