import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
_df = pd.read_csv("teachingratings.csv")

# Basic stats
beauty = _df["beauty"].astype(float)
eval_scores = _df["eval"].astype(float)

# Pearson correlation
corr_r, corr_p = stats.pearsonr(beauty, eval_scores)

# Simple OLS
m_simple = smf.ols("eval ~ beauty", data=_df).fit()

# Full model controls
formula_controls = (
    "eval ~ beauty + age + C(gender) + C(minority) + C(credits) + "
    "C(division) + C(native) + C(tenure) + students + allstudents"
)

m_full = smf.ols(formula_controls, data=_df).fit(cov_type="HC3")

# Also fit non-robust for nested comparison
m_full_nr = smf.ols(formula_controls, data=_df).fit()

formula_controls_no_beauty = (
    "eval ~ age + C(gender) + C(minority) + C(credits) + "
    "C(division) + C(native) + C(tenure) + students + allstudents"
)

m_full_no_beauty = smf.ols(formula_controls_no_beauty, data=_df).fit()

# Standardized effect (beta)
beauty_sd = beauty.std(ddof=1)
eval_sd = eval_scores.std(ddof=1)

beta_simple = m_simple.params["beauty"] * beauty_sd / eval_sd
beta_full = m_full.params["beauty"] * beauty_sd / eval_sd

# 95% CI for beauty coefficient (robust)
conf_full = m_full.conf_int().loc["beauty"].tolist()

# Effect per 1 SD of beauty
sd_effect = m_full.params["beauty"] * beauty_sd

# Partial F-test for beauty in full model (non-robust)
# Use standard ANOVA for nested models
anova_res = sm.stats.anova_lm(m_full_no_beauty, m_full_nr)

# Collect results
results = {
    "n": int(_df.shape[0]),
    "corr_r": float(corr_r),
    "corr_p": float(corr_p),
    "simple_coef": float(m_simple.params["beauty"]),
    "simple_p": float(m_simple.pvalues["beauty"]),
    "simple_r2": float(m_simple.rsquared),
    "full_coef": float(m_full.params["beauty"]),
    "full_p_robust": float(m_full.pvalues["beauty"]),
    "full_conf_int_robust": [float(conf_full[0]), float(conf_full[1])],
    "full_r2": float(m_full.rsquared),
    "full_adj_r2": float(m_full.rsquared_adj),
    "beta_simple": float(beta_simple),
    "beta_full": float(beta_full),
    "beauty_sd": float(beauty_sd),
    "eval_sd": float(eval_sd),
    "sd_effect": float(sd_effect),
    "anova_p": float(anova_res["Pr(>F)"].iloc[1]),
    "anova_f": float(anova_res["F"].iloc[1]),
}

print(json.dumps(results, indent=2))
