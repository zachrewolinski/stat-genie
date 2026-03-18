import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
_df = pd.read_csv('affairs.csv')

# Map columns for clarity
cols = _df.columns.tolist()
# Assume feature2 is affair frequency, feature6 is children yes/no

# Basic cleaning
_df['children'] = _df['feature6'].astype(str).str.lower().map({'yes':1,'no':0})
_df['affair'] = _df['feature2']
_df['any_affair'] = (_df['affair'] > 0).astype(int)

# Drop rows with missing
df = _df.dropna(subset=['children','affair','any_affair'])

# Group stats
summary = df.groupby('children').agg(
    n=('affair','size'),
    mean_affair=('affair','mean'),
    median_affair=('affair','median'),
    prop_any=('any_affair','mean')
)

# Welch t-test for mean affair
affair_yes = df.loc[df['children']==1,'affair']
affair_no = df.loc[df['children']==0,'affair']

t_stat, t_p = stats.ttest_ind(affair_yes, affair_no, equal_var=False, nan_policy='omit')

# Mann-Whitney U test (two-sided)
try:
    mw_stat, mw_p = stats.mannwhitneyu(affair_yes, affair_no, alternative='two-sided')
except ValueError:
    mw_stat, mw_p = (np.nan, np.nan)

# Any-affair proportion difference test (chi-square / z test)
contingency = pd.crosstab(df['children'], df['any_affair'])
# Ensure both categories present
if contingency.shape == (2,2):
    chi2, chi_p, _, _ = stats.chi2_contingency(contingency)
else:
    chi2, chi_p = (np.nan, np.nan)

# Effect size for proportion difference
prop_yes = summary.loc[1,'prop_any']
prop_no = summary.loc[0,'prop_any']
prop_diff = prop_yes - prop_no

# Cohen's d for mean difference
mean_diff = summary.loc[1,'mean_affair'] - summary.loc[0,'mean_affair']
# pooled std (use sample std with ddof=1)
std_yes = affair_yes.std(ddof=1)
std_no = affair_no.std(ddof=1)
pooled = np.sqrt(((len(affair_yes)-1)*std_yes**2 + (len(affair_no)-1)*std_no**2) / (len(affair_yes)+len(affair_no)-2))
cohens_d = mean_diff / pooled if pooled>0 else np.nan

# Regression: OLS on affair score with controls
# Build formula with available covariates
formula = 'affair ~ children'
# Potential controls
controls = ['feature3','feature4','feature5','feature7','feature8','feature9','feature10']
for c in controls:
    if c in df.columns:
        formula += f' + {c}'

# Convert categorical gender
if 'feature3' in df.columns:
    df['feature3'] = df['feature3'].astype('category')

ols_model = smf.ols(formula, data=df).fit(cov_type='HC3')

# Logistic regression for any_affair
logit_formula = 'any_affair ~ children'
for c in controls:
    if c in df.columns:
        logit_formula += f' + {c}'

logit_model = smf.logit(logit_formula, data=df).fit(disp=False)

# Extract key stats
ols_coef = ols_model.params.get('children', np.nan)
ols_p = ols_model.pvalues.get('children', np.nan)

logit_coef = logit_model.params.get('children', np.nan)
logit_p = logit_model.pvalues.get('children', np.nan)
# odds ratio
logit_or = np.exp(logit_coef) if pd.notnull(logit_coef) else np.nan

# Output
print('SUMMARY')
print(summary)
print('\nTTEST', t_stat, t_p)
print('MANNWHITNEY', mw_stat, mw_p)
print('CHI2', chi2, chi_p)
print('MEAN_DIFF', mean_diff)
print('COHENS_D', cohens_d)
print('PROP_DIFF', prop_diff)
print('\nOLS children coef', ols_coef, 'p', ols_p)
print('LOGIT children coef', logit_coef, 'p', logit_p, 'OR', logit_or)
