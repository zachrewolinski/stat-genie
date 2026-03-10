import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "soccer.csv"
df = pd.read_csv(path)

# Ensure numeric
for col in ["feature18", "feature19", "feature16", "feature9"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Average skin rating across raters (0 to 1 scale)
df["skin_avg"] = df[["feature18", "feature19"]].mean(axis=1, skipna=True)

# Keep rows with skin ratings and positive exposure (games)
df = df[df["skin_avg"].notna() & (df["feature9"] > 0)].copy()

# Binary dark vs light (exclude neutral at 0.5)
df["dark"] = df["skin_avg"] > 0.5
df["light"] = df["skin_avg"] < 0.5

df_bin = df[df["dark"] | df["light"]].copy()
df_bin["dark"] = df_bin["dark"].astype(int)

# Group summaries
summary = (
    df_bin.groupby("dark")
    .agg(n=("skin_avg", "size"), red=("feature16", "sum"), games=("feature9", "sum"))
    .assign(rate_per_100=lambda x: 100 * x["red"] / x["games"])
)

# Poisson regression with exposure (games) and robust SE
exog = sm.add_constant(df_bin["dark"])
poisson_model = sm.GLM(
    df_bin["feature16"],
    exog,
    family=sm.families.Poisson(),
    offset=np.log(df_bin["feature9"]),
)
poisson_res = poisson_model.fit(cov_type="HC3")

rr = float(np.exp(poisson_res.params["dark"]))
ci = poisson_res.conf_int().loc["dark"].tolist()
rr_ci = [float(np.exp(ci[0])), float(np.exp(ci[1]))]
p_val = float(poisson_res.pvalues["dark"])

# Adjusted model with league and position controls
# Use formula API for categorical variables
adj_formula = "feature16 ~ dark + C(feature4) + C(feature8)"
adj_model = smf.glm(
    adj_formula,
    data=df_bin,
    family=sm.families.Poisson(),
    offset=np.log(df_bin["feature9"]),
)
adj_res = adj_model.fit(cov_type="HC3")
adj_rr = float(np.exp(adj_res.params["dark"]))
adj_ci = adj_res.conf_int().loc["dark"].tolist()
adj_rr_ci = [float(np.exp(adj_ci[0])), float(np.exp(adj_ci[1]))]
adj_p = float(adj_res.pvalues["dark"])

# Continuous skin tone model (sensitivity)
cont_formula = "feature16 ~ skin_avg"
cont_model = smf.glm(
    cont_formula,
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df["feature9"]),
)
cont_res = cont_model.fit(cov_type="HC3")
cont_rr = float(np.exp(cont_res.params["skin_avg"]))
cont_ci = cont_res.conf_int().loc["skin_avg"].tolist()
cont_rr_ci = [float(np.exp(cont_ci[0])), float(np.exp(cont_ci[1]))]
cont_p = float(cont_res.pvalues["skin_avg"])

# Overdispersion diagnostic (Poisson)
pearson_chi2 = float(poisson_res.pearson_chi2)
df_resid = float(poisson_res.df_resid)
overdispersion = pearson_chi2 / df_resid if df_resid > 0 else float("nan")

results = {
    "n_rows": int(len(df)),
    "n_rows_bin": int(len(df_bin)),
    "summary": {
        "light": {
            "n": int(summary.loc[0, "n"]),
            "red": float(summary.loc[0, "red"]),
            "games": float(summary.loc[0, "games"]),
            "rate_per_100": float(summary.loc[0, "rate_per_100"]),
        },
        "dark": {
            "n": int(summary.loc[1, "n"]),
            "red": float(summary.loc[1, "red"]),
            "games": float(summary.loc[1, "games"]),
            "rate_per_100": float(summary.loc[1, "rate_per_100"]),
        },
    },
    "poisson": {
        "rr": rr,
        "rr_ci": rr_ci,
        "p": p_val,
    },
    "poisson_adjusted": {
        "rr": adj_rr,
        "rr_ci": adj_rr_ci,
        "p": adj_p,
    },
    "poisson_continuous": {
        "rr_per_unit": cont_rr,
        "rr_ci": cont_rr_ci,
        "p": cont_p,
    },
    "overdispersion_ratio": overdispersion,
}

print(json.dumps(results, indent=2))
