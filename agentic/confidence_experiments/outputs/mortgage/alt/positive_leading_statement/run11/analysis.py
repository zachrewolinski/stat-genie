import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data

df = pd.read_csv('mortgage.csv')

# Keep binary vars numeric; avoid forcing int when NaNs exist
binary_cols = ['female','black','self_employed','married','bad_history','deny','accept','denied_PMI']
for c in binary_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Basic counts
n = len(df)

# Denial rate by gender (drop rows with missing female/deny)
df_gender = df[['female','deny']].dropna()
rate_female = df_gender.loc[df_gender['female']==1, 'deny'].mean()
rate_male = df_gender.loc[df_gender['female']==0, 'deny'].mean()

# Two-proportion z-test (female vs male) on denial rate
# Use statsmodels proportion test via scipy? We'll do manual.

count_female = df_gender.loc[df_gender['female']==1, 'deny'].sum()
count_male = df_gender.loc[df_gender['female']==0, 'deny'].sum()

n_female = (df_gender['female']==1).sum()
n_male = (df_gender['female']==0).sum()

# z-test for difference in proportions
p_pool = (count_female + count_male) / (n_female + n_male)
se = np.sqrt(p_pool * (1 - p_pool) * (1/n_female + 1/n_male))
if se == 0:
    z_stat = np.nan
    p_value = np.nan
else:
    z_stat = (count_female/n_female - count_male/n_male) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

# Chi-square test of independence
cont = pd.crosstab(df_gender['female'], df_gender['deny'])
chi2, chi2_p, _, _ = stats.chi2_contingency(cont)

# Logistic regression
# Outcome deny; predictors: female plus key covariates
# Exclude accept (perfect collinear), and maybe exclude denied_PMI because it may be post-decision.
# We'll run two models: baseline (female only) and adjusted.

model1 = smf.logit('deny ~ female', data=df).fit(disp=False)

# Choose covariates: black, housing_expense_ratio, self_employed, married,
# mortgage_credit, consumer_credit, bad_history, PI_ratio, loan_to_value
# We'll exclude denied_PMI and accept.

formula = 'deny ~ female + black + housing_expense_ratio + self_employed + married + mortgage_credit + consumer_credit + bad_history + PI_ratio + loan_to_value'
model2 = smf.logit(formula, data=df).fit(disp=False)

# Extract female coefficient odds ratio and p-value
coef_female_m1 = model1.params['female']
p_female_m1 = model1.pvalues['female']
or_female_m1 = np.exp(coef_female_m1)

coef_female_m2 = model2.params['female']
p_female_m2 = model2.pvalues['female']
or_female_m2 = np.exp(coef_female_m2)


# Save summary results
results = {
    'n': int(n),
    'n_female': int(n_female),
    'n_male': int(n_male),
    'denial_rate_female': float(rate_female),
    'denial_rate_male': float(rate_male),
    'z_stat': float(z_stat) if not np.isnan(z_stat) else None,
    'p_value': float(p_value) if not np.isnan(p_value) else None,
    'chi2_p': float(chi2_p),
    'or_female_m1': float(or_female_m1),
    'p_female_m1': float(p_female_m1),
    'or_female_m2': float(or_female_m2),
    'p_female_m2': float(p_female_m2)
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
