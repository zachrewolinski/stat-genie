import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
with open('info.json', 'r') as f:
    info = json.load(f)

df = pd.read_csv('mortgage.csv')

# Map descriptions
fields = info['data_desc']['fields']
desc_map = {f['column']: (f['properties'].get('description') or '').lower() for f in fields}

# Identify gender column from description
gender_col = None
for col, desc in desc_map.items():
    if 'female' in desc and 'male' in desc:
        gender_col = col
        break
if gender_col is None:
    for col, desc in desc_map.items():
        if 'female' in desc:
            gender_col = col
            break

# Identify approval/denial column from description
approve_col = None
deny_col = None
for col, desc in desc_map.items():
    if 'mortgage application' in desc and 'accepted' in desc and 'denied' in desc:
        pos_accepted = desc.find('accepted')
        pos_denied = desc.find('denied')
        pos_0if = desc.find('0 if')
        # If 'accepted' appears before '0 if', then 1 indicates accepted
        if pos_0if != -1 and pos_accepted != -1 and pos_accepted < pos_0if:
            approve_col = col
        else:
            deny_col = col

if gender_col is None or (approve_col is None and deny_col is None):
    raise RuntimeError('Could not identify gender or approval column from metadata')

if approve_col is not None:
    approve = df[approve_col]
    approve_col_used = approve_col
else:
    approve = 1 - df[deny_col]
    approve_col_used = f"1 - {deny_col}"

# Prepare analysis dataframe
analysis_df = df.copy()
analysis_df['approve'] = approve
analysis_df = analysis_df.replace([np.inf, -np.inf], np.nan)
analysis_df = analysis_df.dropna(subset=[gender_col, 'approve'])

# Crosstab and rates
ct = pd.crosstab(analysis_df[gender_col], analysis_df['approve'])

female_mask = analysis_df[gender_col] == 1
male_mask = analysis_df[gender_col] == 0

female_rate = analysis_df.loc[female_mask, 'approve'].mean()
male_rate = analysis_df.loc[male_mask, 'approve'].mean()

# Difference in proportions (female - male)
# standard error for difference
n_f = female_mask.sum()
n_m = male_mask.sum()

p_f = female_rate
p_m = male_rate
se_diff = np.sqrt(p_f*(1-p_f)/n_f + p_m*(1-p_m)/n_m)
if se_diff > 0:
    z = (p_f - p_m) / se_diff
    p_value_diff = 2 * (1 - stats.norm.cdf(abs(z)))
    ci_low = (p_f - p_m) - 1.96 * se_diff
    ci_high = (p_f - p_m) + 1.96 * se_diff
else:
    z = np.nan
    p_value_diff = np.nan
    ci_low = np.nan
    ci_high = np.nan

# Chi-square test of independence
chi2, chi2_p, dof, expected = stats.chi2_contingency(ct)

# Logistic regression with controls
# Build X with gender and other covariates, drop high-cardinality columns (likely IDs)
X = analysis_df.drop(columns=['approve'])
# Drop columns with too many unique values (likely IDs)
max_unique = int(len(analysis_df) * 0.95)
X = X.loc[:, X.nunique() < max_unique]

# Drop approval column if present
if approve_col in X.columns:
    X = X.drop(columns=[approve_col])

# Ensure gender included
if gender_col not in X.columns:
    X[gender_col] = analysis_df[gender_col]

# Drop rows with missing values in X
X = X.replace([np.inf, -np.inf], np.nan)
valid = X.notna().all(axis=1)
X = X.loc[valid]
y = analysis_df.loc[valid, 'approve']

# Prepare design matrix
X = sm.add_constant(X, has_constant='add')

logit_result = None
logit_error = None
try:
    model = sm.Logit(y, X)
    logit_result = model.fit(disp=False, maxiter=200)
except Exception as e:
    logit_error = str(e)

# Output results
print('gender_col', gender_col)
print('approve_col_used', approve_col_used)
print('n_female', n_f, 'n_male', n_m)
print('female_rate', female_rate)
print('male_rate', male_rate)
print('diff (female - male)', p_f - p_m)
print('diff p-value', p_value_diff)
print('diff 95% CI', ci_low, ci_high)
print('chi2 p-value', chi2_p)

if logit_result is not None:
    coef = logit_result.params[gender_col]
    pval = logit_result.pvalues[gender_col]
    oratio = float(np.exp(coef))
    print('logit coef', coef, 'pval', pval, 'odds_ratio', oratio)
else:
    print('logit_error', logit_error)
