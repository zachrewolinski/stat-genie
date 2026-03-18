import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
file_path = 'affairs.csv'
df = pd.read_csv(file_path)

# Identify columns
# feature2: affair frequency
# feature6: children yes/no

# Clean/prepare
# Ensure consistent casing
df['feature6'] = df['feature6'].astype(str).str.strip().str.lower()

# Binary indicator: children yes=1, no=0
child_map = {'yes': 1, 'no': 0}
df['children'] = df['feature6'].map(child_map)

# Drop rows with missing children or affair data
analysis_df = df[['feature2','children','feature3','feature4','feature5','feature7','feature8','feature9','feature10']].dropna()

# Basic group stats
by_child = analysis_df.groupby('children')['feature2']
summary = by_child.agg(['count','mean','median','std'])

# Welch t-test
child_yes = analysis_df[analysis_df['children'] == 1]['feature2']
child_no = analysis_df[analysis_df['children'] == 0]['feature2']

welch_t = stats.ttest_ind(child_yes, child_no, equal_var=False)

# Mann-Whitney U test (two-sided)
# Use alternative 'two-sided' if available
try:
    mwu = stats.mannwhitneyu(child_yes, child_no, alternative='two-sided')
except TypeError:
    # older scipy fallback
    mwu = stats.mannwhitneyu(child_yes, child_no)

# Effect size: Cohen's d (using pooled SD for independent groups)
mean_yes = child_yes.mean()
mean_no = child_no.mean()
std_yes = child_yes.std(ddof=1)
std_no = child_no.std(ddof=1)

n_yes = child_yes.shape[0]
n_no = child_no.shape[0]

pooled_sd = np.sqrt(((n_yes-1)*std_yes**2 + (n_no-1)*std_no**2) / (n_yes+n_no-2))
cohens_d = (mean_yes - mean_no) / pooled_sd if pooled_sd != 0 else np.nan

# Proportion with any affairs
analysis_df['any_affair'] = (analysis_df['feature2'] > 0).astype(int)

cont_table = pd.crosstab(analysis_df['children'], analysis_df['any_affair'])
chi2, p_chi2, dof, expected = stats.chi2_contingency(cont_table)

# Logistic regression: any affair ~ children + controls
# Prepare design matrix
X = analysis_df[['children','feature3','feature4','feature5','feature7','feature8','feature9','feature10']].copy()

# Encode gender (feature3) as binary: female=1, male=0
X['feature3'] = X['feature3'].astype(str).str.strip().str.lower()
X['female'] = (X['feature3'] == 'female').astype(int)

# Drop original gender
X = X.drop(columns=['feature3'])

# Add constant
X = sm.add_constant(X, has_constant='add')

y = analysis_df['any_affair']

logit_model = sm.Logit(y, X, missing='drop')
logit_result = logit_model.fit(disp=0)

# Extract child coefficient and p-value
child_coef = logit_result.params['children']
child_p = logit_result.pvalues['children']

# OLS regression on affair frequency (feature2) with controls
# Note: feature2 is skewed, but OLS used for directionality
X2 = analysis_df[['children','feature3','feature4','feature5','feature7','feature8','feature9','feature10']].copy()
X2['feature3'] = X2['feature3'].astype(str).str.strip().str.lower()
X2['female'] = (X2['feature3'] == 'female').astype(int)
X2 = X2.drop(columns=['feature3'])
X2 = sm.add_constant(X2, has_constant='add')

ols_model = sm.OLS(analysis_df['feature2'], X2)
ols_result = ols_model.fit()

child_coef_ols = ols_result.params['children']
child_p_ols = ols_result.pvalues['children']

# Save results summary to JSON-like dict for easy reading
results = {
    'summary': summary.to_dict(),
    'welch_t_stat': float(welch_t.statistic),
    'welch_t_p': float(welch_t.pvalue),
    'mannwhitney_u': float(mwu.statistic),
    'mannwhitney_p': float(mwu.pvalue),
    'cohens_d': float(cohens_d),
    'cont_table': cont_table.to_dict(),
    'chi2': float(chi2),
    'chi2_p': float(p_chi2),
    'logit_child_coef': float(child_coef),
    'logit_child_p': float(child_p),
    'ols_child_coef': float(child_coef_ols),
    'ols_child_p': float(child_p_ols),
}

# Print results
print('Group summary:')
print(summary)
print('\nWelch t-test:', welch_t)
print('Mann-Whitney U:', mwu)
print('Cohen d:', cohens_d)
print('\nAny affair contingency table:')
print(cont_table)
print('Chi-square p:', p_chi2)
print('\nLogit child coef:', child_coef, 'p:', child_p)
print('OLS child coef:', child_coef_ols, 'p:', child_p_ols)
