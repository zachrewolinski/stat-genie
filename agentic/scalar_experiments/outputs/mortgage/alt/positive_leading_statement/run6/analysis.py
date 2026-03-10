import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
from scipy import stats


df = pd.read_csv('mortgage.csv')

# Basic cleaning
# Coerce to numeric and keep missing as NaN; drop rows with missing outcome/exposure
for col in ['female', 'accept', 'deny']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=['female', 'accept'])
df['female'] = df['female'].astype(int)
df['accept'] = df['accept'].astype(int)
if 'deny' in df.columns:
    df['deny'] = df['deny'].astype(int)

# Use accept as outcome (1 accepted, 0 denied)
# Unadjusted approval rates by gender
rates = df.groupby('female')['accept'].agg(['mean', 'count', 'sum']).rename(columns={'mean':'accept_rate', 'sum':'accepted'})

# Two-proportion z-test for difference in acceptance rates
count = rates['accepted'].values
nobs = rates['count'].values
zstat, pval = proportions_ztest(count, nobs)

# Chi-square test of independence (female vs accept)
contingency = pd.crosstab(df['female'], df['accept'])
chi2, chi_p, dof, exp = stats.chi2_contingency(contingency)

# Logistic regression: unadjusted
X = sm.add_constant(df['female'])
model_unadj = sm.Logit(df['accept'], X).fit(disp=False)

# Adjusted model with standard creditworthiness controls (exclude clearly post-decision variables)
controls = ['black', 'housing_expense_ratio', 'self_employed', 'married',
            'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio', 'loan_to_value']
controls = [c for c in controls if c in df.columns]
df_adj = df[['accept', 'female'] + controls].dropna()
X_adj = sm.add_constant(df_adj[['female'] + controls])
model_adj = sm.Logit(df_adj['accept'], X_adj).fit(disp=False)

# Average marginal effect (predictive difference) for female in adjusted model
X0 = X_adj.copy()
X1 = X_adj.copy()
X0['female'] = 0
X1['female'] = 1
pred0 = model_adj.predict(X0)
pred1 = model_adj.predict(X1)
ame = (pred1 - pred0).mean()

# Odds ratios and CI for female
def odds_ratio_ci(model, var='female'):
    coef = model.params[var]
    se = model.bse[var]
    or_val = np.exp(coef)
    ci_low = np.exp(coef - 1.96*se)
    ci_high = np.exp(coef + 1.96*se)
    p = model.pvalues[var]
    return or_val, ci_low, ci_high, p

or_unadj = odds_ratio_ci(model_unadj, 'female')
or_adj = odds_ratio_ci(model_adj, 'female')

# Print results
print('Counts/accept rates by gender (female=1):')
print(rates)
print('\nTwo-proportion z-test (acceptance rate difference): z=%.4f, p=%.6f' % (zstat, pval))
print('Chi-square test: chi2=%.4f, p=%.6f, dof=%d' % (chi2, chi_p, dof))
print('\nLogit unadjusted for female:')
print(model_unadj.summary().tables[1])
print('OR (female): %.4f, 95%% CI [%.4f, %.4f], p=%.6f' % or_unadj)
print('\nLogit adjusted for female + controls:')
coef_f = model_adj.params['female']
se_f = model_adj.bse['female']
p_f = model_adj.pvalues['female']
print('female coef=%.4f, SE=%.4f, z=%.4f, p=%.6f' % (coef_f, se_f, coef_f / se_f, p_f))
print('OR (female): %.4f, 95%% CI [%.4f, %.4f], p=%.6f' % or_adj)
print('Adjusted average marginal effect (female=1 vs 0): %.4f' % ame)
