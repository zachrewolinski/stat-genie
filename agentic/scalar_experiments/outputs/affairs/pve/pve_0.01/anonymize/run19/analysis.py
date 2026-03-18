import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

DATA_PATH = "affairs.csv"

df = pd.read_csv(DATA_PATH)

# Basic checks
n_total = len(df)

# Map children indicator
children = df["feature6"].astype(str).str.lower()

# Outcome: affair frequency numeric
affairs = df["feature2"].astype(float)

# Groups
mask_yes = children == "yes"
mask_no = children == "no"

# Group stats
summary = {
    "n_total": int(n_total),
    "n_yes": int(mask_yes.sum()),
    "n_no": int(mask_no.sum()),
    "mean_yes": float(affairs[mask_yes].mean()),
    "mean_no": float(affairs[mask_no].mean()),
    "median_yes": float(affairs[mask_yes].median()),
    "median_no": float(affairs[mask_no].median()),
    "std_yes": float(affairs[mask_yes].std(ddof=1)),
    "std_no": float(affairs[mask_no].std(ddof=1)),
}

# Two-sample t-test (Welch)
t_stat, t_p = stats.ttest_ind(affairs[mask_yes], affairs[mask_no], equal_var=False, nan_policy="omit")

# Mann-Whitney U
u_stat, u_p = stats.mannwhitneyu(affairs[mask_yes], affairs[mask_no], alternative="two-sided")

summary.update({
    "t_stat": float(t_stat),
    "t_p": float(t_p),
    "u_stat": float(u_stat),
    "u_p": float(u_p),
})

# Effect size: Cohen's d (using pooled SD)
mean_yes = affairs[mask_yes].mean()
mean_no = affairs[mask_no].mean()
std_yes = affairs[mask_yes].std(ddof=1)
std_no = affairs[mask_no].std(ddof=1)

n_yes = mask_yes.sum()
n_no = mask_no.sum()

pooled_sd = np.sqrt(((n_yes - 1) * std_yes ** 2 + (n_no - 1) * std_no ** 2) / (n_yes + n_no - 2))
cohen_d = (mean_yes - mean_no) / pooled_sd if pooled_sd > 0 else np.nan
summary["cohen_d_yes_minus_no"] = float(cohen_d)

# Any-affair indicator
any_affair = (affairs > 0).astype(int)

# Proportions
prop_yes = any_affair[mask_yes].mean()
prop_no = any_affair[mask_no].mean()

summary.update({
    "prop_any_yes": float(prop_yes),
    "prop_any_no": float(prop_no),
})

# Two-proportion z-test
count = np.array([any_affair[mask_yes].sum(), any_affair[mask_no].sum()])
obs = np.array([mask_yes.sum(), mask_no.sum()])

# manual z-test
p_pool = count.sum() / obs.sum()
se = np.sqrt(p_pool * (1 - p_pool) * (1/obs[0] + 1/obs[1]))
if se > 0:
    z_stat = (prop_yes - prop_no) / se
    z_p = 2 * (1 - stats.norm.cdf(abs(z_stat)))
else:
    z_stat = np.nan
    z_p = np.nan
summary.update({
    "z_stat": float(z_stat),
    "z_p": float(z_p),
})

# Logistic regression: any_affair ~ children + controls
# Controls: gender(feature3), age(feature4), years married(feature5), religiousness(feature7), education(feature8), occupation(feature9), marriage rating(feature10)
# Build design matrix
X = pd.DataFrame({
    "children_yes": (children == "yes").astype(int),
    "gender_male": (df["feature3"].astype(str).str.lower() == "male").astype(int),
    "age": df["feature4"].astype(float),
    "years_married": df["feature5"].astype(float),
    "religiousness": df["feature7"].astype(float),
    "education": df["feature8"].astype(float),
    "occupation": df["feature9"].astype(float),
    "marriage_rating": df["feature10"].astype(float),
})
X = sm.add_constant(X, has_constant="add")

logit_model = sm.Logit(any_affair, X)
try:
    logit_res = logit_model.fit(disp=False)
    coef = logit_res.params["children_yes"]
    se_coef = logit_res.bse["children_yes"]
    p_coef = logit_res.pvalues["children_yes"]
    summary.update({
        "logit_coef_children_yes": float(coef),
        "logit_se_children_yes": float(se_coef),
        "logit_p_children_yes": float(p_coef),
        "logit_odds_ratio_children_yes": float(np.exp(coef)),
    })
except Exception as e:
    summary.update({
        "logit_error": str(e)
    })

# OLS on log(1+affairs)
log_affairs = np.log1p(affairs)
ols_model = sm.OLS(log_affairs, X)
try:
    ols_res = ols_model.fit()
    coef = ols_res.params["children_yes"]
    se_coef = ols_res.bse["children_yes"]
    p_coef = ols_res.pvalues["children_yes"]
    summary.update({
        "ols_coef_children_yes": float(coef),
        "ols_se_children_yes": float(se_coef),
        "ols_p_children_yes": float(p_coef),
    })
except Exception as e:
    summary.update({
        "ols_error": str(e)
    })

print(json.dumps(summary, indent=2))
