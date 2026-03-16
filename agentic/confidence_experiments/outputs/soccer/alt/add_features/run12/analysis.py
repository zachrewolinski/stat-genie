import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
path = "soccer.csv"
df = pd.read_csv(path)

# Basic cleaning
# Ensure numeric columns
for col in ["rater1", "rater2", "redCards", "games"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Compute mean skin tone rating
skin_mean = df[["rater1", "rater2"]].mean(axis=1, skipna=True)

# Define light/dark groups based on 5-point scale normalized to [0,1]
# Very light/light <= 0.25; dark/very dark >= 0.75
light_mask = skin_mean <= 0.25

dark_mask = skin_mean >= 0.75

# Subset for analysis
sub = df[light_mask | dark_mask].copy()
sub["skin_mean"] = skin_mean[light_mask | dark_mask]
sub["dark"] = (dark_mask[light_mask | dark_mask]).astype(int)

# Exclude invalid games
sub = sub[sub["games"].notna() & (sub["games"] > 0) & sub["redCards"].notna()]

# Summary stats
summary = {}
summary["n_total"] = int(len(sub))
summary["n_light"] = int((sub["dark"] == 0).sum())
summary["n_dark"] = int((sub["dark"] == 1).sum())

# Rates
sub["red_rate"] = sub["redCards"] / sub["games"]
summary["mean_red_rate_light"] = float(sub.loc[sub["dark"] == 0, "red_rate"].mean())
summary["mean_red_rate_dark"] = float(sub.loc[sub["dark"] == 1, "red_rate"].mean())
summary["mean_red_any_light"] = float((sub.loc[sub["dark"] == 0, "redCards"] > 0).mean())
summary["mean_red_any_dark"] = float((sub.loc[sub["dark"] == 1, "redCards"] > 0).mean())

# Poisson regression with offset log(games)
X = sm.add_constant(sub[["dark"]])
offset = np.log(sub["games"].values)
poisson_model = sm.GLM(sub["redCards"], X, family=sm.families.Poisson(), offset=offset)
poisson_res = poisson_model.fit(cov_type="HC1")

coef = poisson_res.params["dark"]
se = poisson_res.bse["dark"]
pval = poisson_res.pvalues["dark"]
rr = float(np.exp(coef))
ci_low, ci_high = poisson_res.conf_int().loc["dark"]
rr_ci = (float(np.exp(ci_low)), float(np.exp(ci_high)))

summary["poisson_coef_dark"] = float(coef)
summary["poisson_se_dark"] = float(se)
summary["poisson_pval_dark"] = float(pval)
summary["poisson_rr_dark"] = rr
summary["poisson_rr_ci"] = rr_ci

# Logistic regression for any red card, controlling for log(games)
sub["red_any"] = (sub["redCards"] > 0).astype(int)
X_logit = sm.add_constant(sub[["dark"]])
X_logit["log_games"] = np.log(sub["games"].values)
logit_model = sm.Logit(sub["red_any"], X_logit)
logit_res = logit_model.fit(disp=False)

coef_l = logit_res.params["dark"]
se_l = logit_res.bse["dark"]
pval_l = logit_res.pvalues["dark"]
or_l = float(np.exp(coef_l))
ci_low_l, ci_high_l = logit_res.conf_int().loc["dark"]
or_ci = (float(np.exp(ci_low_l)), float(np.exp(ci_high_l)))

summary["logit_coef_dark"] = float(coef_l)
summary["logit_se_dark"] = float(se_l)
summary["logit_pval_dark"] = float(pval_l)
summary["logit_or_dark"] = or_l
summary["logit_or_ci"] = or_ci

# Continuous skin tone sensitivity using Poisson on all with skin_mean
sub_cont = df.copy()
sub_cont["skin_mean"] = skin_mean
sub_cont = sub_cont[sub_cont["skin_mean"].notna() & sub_cont["games"].notna() & (sub_cont["games"] > 0) & sub_cont["redCards"].notna()]

X_cont = sm.add_constant(sub_cont[["skin_mean"]])
offset_cont = np.log(sub_cont["games"].values)
poisson_cont = sm.GLM(sub_cont["redCards"], X_cont, family=sm.families.Poisson(), offset=offset_cont)
poisson_cont_res = poisson_cont.fit(cov_type="HC1")

coef_c = poisson_cont_res.params["skin_mean"]
se_c = poisson_cont_res.bse["skin_mean"]
pval_c = poisson_cont_res.pvalues["skin_mean"]
rr_c = float(np.exp(coef_c))
ci_low_c, ci_high_c = poisson_cont_res.conf_int().loc["skin_mean"]
rr_ci_c = (float(np.exp(ci_low_c)), float(np.exp(ci_high_c)))

summary["poisson_cont_coef"] = float(coef_c)
summary["poisson_cont_se"] = float(se_c)
summary["poisson_cont_pval"] = float(pval_c)
summary["poisson_cont_rr"] = rr_c
summary["poisson_cont_rr_ci"] = rr_ci_c

print(json.dumps(summary, indent=2))
