import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
path = "affairs.csv"
df = pd.read_csv(path)

# Keep relevant columns
# children is categorical yes/no; affairs numeric
# Some columns may have missing values; drop rows with missing on key vars
key_cols = ["affairs", "children", "gender", "age", "yearsmarried", "religiousness", "education", "occupation", "rating"]
df_key = df[key_cols].dropna().copy()

# Normalize children to lower-case strings
if df_key["children"].dtype.name == "category":
    df_key["children"] = df_key["children"].astype(str)

# Ensure consistent labels
children = df_key["children"].str.lower().str.strip()
df_key["children"] = children

# Split groups
grp_yes = df_key[df_key["children"] == "yes"]["affairs"]
grp_no = df_key[df_key["children"] == "no"]["affairs"]

# Basic stats
mean_yes = grp_yes.mean()
mean_no = grp_no.mean()
std_yes = grp_yes.std(ddof=1)
std_no = grp_no.std(ddof=1)

# Pooled std for Cohen's d
n_yes = grp_yes.shape[0]
n_no = grp_no.shape[0]
pooled_var = ((n_yes - 1) * std_yes**2 + (n_no - 1) * std_no**2) / (n_yes + n_no - 2)
pooled_sd = np.sqrt(pooled_var)
cohens_d = (mean_no - mean_yes) / pooled_sd if pooled_sd > 0 else 0.0

# t-test for difference in means
# Use Welch t-test
if n_yes > 1 and n_no > 1:
    t_stat, p_val = stats.ttest_ind(grp_no, grp_yes, equal_var=False)
else:
    p_val = 1.0

# Logistic: any affairs > 0
any_affairs = (df_key["affairs"] > 0).astype(int)
df_key["any_affairs"] = any_affairs

# Regression (OLS) for affairs count, with controls
# children treated as categorical
ols_model = smf.ols(
    "affairs ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating",
    data=df_key,
).fit()

# Logistic regression for any affair
logit_model = smf.logit(
    "any_affairs ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating",
    data=df_key,
).fit(disp=False)

# Extract children coefficient (yes vs no)
# With C(children), baseline is alphabetically first: 'no' vs 'yes'.
# Statsmodels uses 'no' as reference if it sorts, but to be safe find the key
ols_coef = 0.0
logit_coef = 0.0
for key in ols_model.params.index:
    if "C(children)" in key:
        ols_coef = ols_model.params[key]
        break
for key in logit_model.params.index:
    if "C(children)" in key:
        logit_coef = logit_model.params[key]
        break

# Score mapping
# Positive score means evidence that children decrease affairs
# Use Cohen's d and p-value to scale confidence
magnitude = 100 * np.tanh(abs(cohens_d))
confidence = max(0.0, 1.0 - min(1.0, p_val))
raw_score = magnitude * confidence
score = int(round(raw_score))

# Determine sign from mean difference (mean_no - mean_yes)
if mean_no < mean_yes:
    score = -score

# If regression coefficients suggest opposite direction than mean difference, nudge toward 0
# OLS coefficient negative (children yes lower affairs) supports positive score
# If sign mismatch, reduce magnitude
ols_support = -1 if ols_coef > 0 else (1 if ols_coef < 0 else 0)
# score>0 means children decrease; score<0 means increase
score_sign = 1 if score > 0 else (-1 if score < 0 else 0)
if ols_support != 0 and score_sign != 0 and ols_support != score_sign:
    score = int(round(score * 0.5))

# Clamp to [-100, 100]
score = max(-100, min(100, score))

# Write conclusion
with open("conclusion.txt", "w") as f:
    f.write(str(score))

# Also save a brief analysis summary for debugging (optional)
summary = {
    "n_yes": int(n_yes),
    "n_no": int(n_no),
    "mean_yes": float(mean_yes),
    "mean_no": float(mean_no),
    "cohens_d": float(cohens_d),
    "p_val": float(p_val),
    "ols_coef": float(ols_coef),
    "logit_coef": float(logit_coef),
    "score": int(score),
}

with open("analysis_summary.json", "w") as f:
    import json
    json.dump(summary, f, indent=2)
