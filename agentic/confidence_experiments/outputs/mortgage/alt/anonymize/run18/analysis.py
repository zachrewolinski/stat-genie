import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data

df = pd.read_csv('mortgage.csv')

# Replace inf with NaN

df = df.replace([np.inf, -np.inf], np.nan)

# Define variables
female = df['feature2']
accepted = df['feature14']

# Basic counts (drop missing for crosstab)
ct = pd.crosstab(female, accepted)

# Acceptance rates by gender
rates = df.groupby('feature2')['feature14'].mean()

# Chi-square test for association (drop rows with missing in either)
chi_df = df[['feature2', 'feature14']].dropna()
ct_chi = pd.crosstab(chi_df['feature2'], chi_df['feature14'])
chi2, p, dof, expected = stats.chi2_contingency(ct_chi)

# Logistic regression: acceptance on gender only
simple_df = df[['feature2', 'feature14']].dropna()
X_simple = sm.add_constant(simple_df['feature2'])
model_simple = sm.Logit(simple_df['feature14'], X_simple).fit(disp=False)

# Logistic regression with controls
# Exclude outcome feature11 and feature14
control_cols = [c for c in df.columns if c not in ['feature14', 'feature11']]
full_df = df[control_cols + ['feature14']].dropna()
X = sm.add_constant(full_df[control_cols])
model_full = sm.Logit(full_df['feature14'], X).fit(disp=False)


def odds_ratio_summary(model, var_name):
    coef = model.params[var_name]
    se = model.bse[var_name]
    z = coef / se
    pval = model.pvalues[var_name]
    # 95% CI
    ci_low = coef - 1.96 * se
    ci_high = coef + 1.96 * se
    return {
        'coef': coef,
        'se': se,
        'z': z,
        'p': pval,
        'or': np.exp(coef),
        'or_ci_low': np.exp(ci_low),
        'or_ci_high': np.exp(ci_high)
    }

summary_simple = odds_ratio_summary(model_simple, 'feature2')
summary_full = odds_ratio_summary(model_full, 'feature2')

# Average marginal effect for female in full model
marginal = model_full.get_margeff(at='overall').summary_frame()
me_female = marginal.loc['feature2']

# Print results
print('Rows total:', len(df))
print('Rows used (simple):', len(simple_df))
print('Rows used (full):', len(full_df))
print('\nCounts (female=0/1 by accepted=0/1):')
print(ct)
print('\nAcceptance rates by gender (feature2=0 male, 1 female):')
print(rates)
print('\nChi-square test: chi2=%.4f, p=%.6f, dof=%d' % (chi2, p, dof))
print('\nLogit simple (accept ~ female):')
print(summary_simple)
print('\nLogit full (accept ~ female + controls):')
print(summary_full)
print('\nMarginal effect for female (full model):')
print(me_female)
