import json
import math
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
DATA_PATH = "panda_nuts.csv"

df = pd.read_csv(DATA_PATH)

# Keep relevant columns and drop missing
cols = ["nuts_opened", "seconds", "age", "sex", "help"]
missing_cols = [c for c in cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

sub = df[cols].copy()
sub = sub.dropna()

# Ensure positive seconds for log offset
sub = sub[sub["seconds"] > 0].copy()

# Set categories with explicit baselines
sub["sex"] = pd.Categorical(sub["sex"].astype(str), categories=["m", "f"], ordered=False)
sub["help"] = pd.Categorical(sub["help"].astype(str), categories=["N", "y"], ordered=False)

# Remove any rows with categories outside expected
sub = sub[sub["sex"].isin(["m", "f"]) & sub["help"].isin(["N", "y"])].copy()

# Poisson GLM with offset for exposure time
formula = "nuts_opened ~ age + C(sex) + C(help)"
model = smf.glm(
    formula=formula,
    data=sub,
    family=sm.families.Poisson(),
    offset=np.log(sub["seconds"])
)
result = model.fit(cov_type="HC0")

# Calculate dispersion (deviance / df_resid)
try:
    dispersion = result.deviance / result.df_resid
except ZeroDivisionError:
    dispersion = float('nan')

# Extract coefficients
params = result.params
conf = result.conf_int()
conf.columns = ["ci_low", "ci_high"]

summary_rows = []
for name in params.index:
    coef = params[name]
    se = result.bse[name]
    pval = result.pvalues[name]
    ci_low, ci_high = conf.loc[name, "ci_low"], conf.loc[name, "ci_high"]
    rr = math.exp(coef)
    rr_low = math.exp(ci_low)
    rr_high = math.exp(ci_high)
    summary_rows.append({
        "term": name,
        "coef": coef,
        "se": se,
        "pval": pval,
        "rr": rr,
        "rr_ci_low": rr_low,
        "rr_ci_high": rr_high,
    })

summary_df = pd.DataFrame(summary_rows)

# Also compute simple descriptive efficiency
sub["rate"] = sub["nuts_opened"] / sub["seconds"]
rate_summary = sub.groupby(["sex", "help"], observed=True)["rate"].agg(["mean", "median", "count"]).reset_index()

output = {
    "n_rows": int(sub.shape[0]),
    "dispersion": float(dispersion),
    "glm_summary": summary_df.to_dict(orient="records"),
    "rate_summary": rate_summary.to_dict(orient="records"),
}

# Negative Binomial (if supported) to address overdispersion
nb_output = None
try:
    nb_model = smf.negativebinomial(
        formula=formula,
        data=sub,
        offset=np.log(sub["seconds"])
    )
    nb_result = nb_model.fit(disp=0)
    nb_params = nb_result.params
    nb_conf = nb_result.conf_int()
    nb_conf.columns = ["ci_low", "ci_high"]
    nb_rows = []
    for name in nb_params.index:
        coef = nb_params[name]
        pval = nb_result.pvalues[name]
        ci_low, ci_high = nb_conf.loc[name, "ci_low"], nb_conf.loc[name, "ci_high"]
        rr = math.exp(coef)
        rr_low = math.exp(ci_low)
        rr_high = math.exp(ci_high)
        nb_rows.append({
            "term": name,
            "coef": coef,
            "pval": pval,
            "rr": rr,
            "rr_ci_low": rr_low,
            "rr_ci_high": rr_high,
        })
    nb_output = nb_rows
except Exception as e:
    nb_output = {"error": str(e)}

output["nb_summary"] = nb_output

with open("analysis_output.json", "w") as f:
    json.dump(output, f, indent=2)

print("Wrote analysis_output.json")
