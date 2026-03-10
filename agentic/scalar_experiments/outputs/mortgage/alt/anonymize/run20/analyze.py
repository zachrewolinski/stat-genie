import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

DATA_PATH = 'mortgage.csv'

# Load and keep relevant columns
raw = pd.read_csv(DATA_PATH)

# Ensure only valid rows
cols = ['feature2', 'feature14']
df = raw[cols].dropna()

female = df['feature2']
accepted = df['feature14']

# contingency table
ct = pd.crosstab(female, accepted)  # rows: female(0/1), cols: accepted(0/1)

# acceptance rates
accept_rate = df.groupby('feature2')['feature14'].mean()

# difference female - male
rate_female = accept_rate.get(1, np.nan)
rate_male = accept_rate.get(0, np.nan)
rate_diff = rate_female - rate_male

# chi-square test
chi2, p_chi, dof, expected = stats.chi2_contingency(ct)

# odds ratio from contingency table
ct = ct.reindex(index=[0,1], columns=[0,1])
# add 0.5 to avoid zero counts if any
odds_male = (ct.loc[0,1] + 0.5) / (ct.loc[0,0] + 0.5)
odds_female = (ct.loc[1,1] + 0.5) / (ct.loc[1,0] + 0.5)
odds_ratio = odds_female / odds_male

# logistic regression (accepted ~ female)
X = sm.add_constant(female)
model = sm.Logit(accepted, X).fit(disp=False)
coef = model.params['feature2']
se = model.bse['feature2']
# Wald p-value
p_logit = model.pvalues['feature2']
# odds ratio and 95% CI
or_logit = np.exp(coef)
ci_low = np.exp(coef - 1.96 * se)
ci_high = np.exp(coef + 1.96 * se)

# sample sizes
n_total = len(df)
counts = df['feature2'].value_counts().to_dict()

print('n_total', n_total)
print('counts', counts)
print('accept_rate', accept_rate.to_dict())
print('rate_diff_female_minus_male', rate_diff)
print('chi2', chi2, 'p', p_chi)
print('odds_ratio (approx)', odds_ratio)
print('logit coef', coef, 'p', p_logit, 'OR', or_logit, 'CI', ci_low, ci_high)
