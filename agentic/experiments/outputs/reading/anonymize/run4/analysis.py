import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('reading.csv')

# Define columns
READER_VIEW = 'feature3'
DYSLEXIA = 'feature17'
READING_SPEED = 'feature20'  # words per minute

# Subset to dyslexic readers
dys = df[df[DYSLEXIA] == 1].copy()

# Basic group stats
rv_on = dys[dys[READER_VIEW] == 1][READING_SPEED]
rv_off = dys[dys[READER_VIEW] == 0][READING_SPEED]

summary = {
    'n_dys_total': len(dys),
    'n_rv_on': rv_on.shape[0],
    'n_rv_off': rv_off.shape[0],
    'mean_rv_on': rv_on.mean(),
    'mean_rv_off': rv_off.mean(),
    'diff_on_minus_off': rv_on.mean() - rv_off.mean(),
    'median_rv_on': rv_on.median(),
    'median_rv_off': rv_off.median(),
}

# Welch t-test (unequal variances)
t_stat, p_val = stats.ttest_ind(rv_on, rv_off, equal_var=False, nan_policy='omit')

# 95% CI for difference in means (Welch)
# Compute using standard error of difference
se_diff = np.sqrt(rv_on.var(ddof=1)/rv_on.shape[0] + rv_off.var(ddof=1)/rv_off.shape[0])
# Welch-Satterthwaite df
num = (rv_on.var(ddof=1)/rv_on.shape[0] + rv_off.var(ddof=1)/rv_off.shape[0])**2
den = ((rv_on.var(ddof=1)/rv_on.shape[0])**2/(rv_on.shape[0]-1)) + ((rv_off.var(ddof=1)/rv_off.shape[0])**2/(rv_off.shape[0]-1))
df_welch = num/den
ci_low, ci_high = stats.t.interval(0.95, df_welch, loc=summary['diff_on_minus_off'], scale=se_diff)

# Regression with basic controls to check robustness
# Use reading speed as outcome; include readability, device, language, age, education, gender, retake, native speaker
# Keep within dyslexic group to answer question
reg_df = dys.copy()
# Convert to categorical for statsmodels
categoricals = ['feature11', 'feature13', 'feature14', 'feature15', 'feature18']
for c in categoricals:
    reg_df[c] = reg_df[c].astype('category')

formula = (
    "feature20 ~ feature3 + feature19 + feature10 + feature16 + C(feature11) + "
    "C(feature13) + C(feature14) + C(feature15) + C(feature18)"
)
model = smf.ols(formula, data=reg_df).fit()

# Collect key regression result
coef = model.params.get('feature3', np.nan)
coef_p = model.pvalues.get('feature3', np.nan)

print('SUMMARY')
for k, v in summary.items():
    print(f"{k}: {v}")
print(f"t_stat: {t_stat}")
print(f"p_val: {p_val}")
print(f"95% CI diff (on - off): [{ci_low}, {ci_high}]")
print(f"reg_coef_feature3: {coef}")
print(f"reg_pval_feature3: {coef_p}")
