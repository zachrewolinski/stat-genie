import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data and metadata
df = pd.read_csv('mortgage.csv')
with open('info.json', 'r') as f:
    info = json.load(f)

# Map columns based on description keywords when available
gender_col = None
deny_col = None
accept_col = None
for field in info.get('data_desc', {}).get('fields', []):
    desc = (field.get('properties', {}).get('description') or '').lower()
    col = field.get('column')
    if 'female' in desc:
        gender_col = col
    if 'application was denied' in desc:
        deny_col = col
    if 'application was accepted' in desc:
        accept_col = col

# Basic info
cols = df.columns.tolist()

# Identify outcome columns from metadata when possible
has_deny = deny_col in df.columns if deny_col else False
has_accept = accept_col in df.columns if accept_col else False

# Check complementarity if both exist
complement = None
if has_deny and has_accept:
    s = df[deny_col] + df[accept_col]
    complement = bool(s.isin([1]).all())

# Choose outcome based on metadata
if has_accept:
    outcome = accept_col
    outcome_positive = 1  # 1 means accepted
    outcome_is_denial = False
elif has_deny:
    outcome = deny_col
    outcome_positive = 1  # 1 means denied
    outcome_is_denial = True
else:
    # Fallback: use a column literally named 'deny' or 'accept' if metadata missing
    if 'deny' in df.columns:
        outcome = 'deny'
        outcome_positive = 1
        outcome_is_denial = True
    elif 'accept' in df.columns:
        outcome = 'accept'
        outcome_positive = 1
        outcome_is_denial = False
    else:
        raise ValueError('No outcome column found.')

# Gender column
if gender_col is None or gender_col not in df.columns:
    if 'female' in df.columns:
        gender_col = 'female'
    else:
        raise ValueError('Gender column not found.')

# Drop missing
analysis_df = df.copy()
analysis_df = analysis_df.dropna(subset=[gender_col, outcome])

# Ensure binary
# Compute approval rate: if outcome is denial, approval = 1 - denial; else approval = acceptance
if outcome_is_denial:
    analysis_df['approved'] = 1 - analysis_df[outcome]
else:
    analysis_df['approved'] = analysis_df[outcome]

# Crosstab
ct = pd.crosstab(analysis_df[gender_col], analysis_df['approved'])

# Approval rates by gender
approval_rates = analysis_df.groupby(gender_col)['approved'].mean()

# Two-proportion z-test for approval rates (female vs male)
# female==1, male==0
f = analysis_df[analysis_df[gender_col] == 1]
m = analysis_df[analysis_df[gender_col] == 0]

n1, n2 = len(f), len(m)
if n1 > 0 and n2 > 0:
    p1, p2 = f['approved'].mean(), m['approved'].mean()
    p_pool = (f['approved'].sum() + m['approved'].sum()) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z = (p1 - p2) / se if se > 0 else np.nan
    p_value = 2 * (1 - stats.norm.cdf(abs(z))) if se > 0 else np.nan
else:
    p1 = p2 = z = p_value = np.nan

# Chi-square test of independence
chi2 = stats.chi2_contingency(ct)
chi2_stat, chi2_p = chi2[0], chi2[1]

# Logistic regression (unadjusted and adjusted)
y = analysis_df[outcome]

# Unadjusted model: gender only
X_unadj = sm.add_constant(analysis_df[[gender_col]], has_constant='add')
logit_unadj = None
try:
    logit_unadj = sm.GLM(y, X_unadj, family=sm.families.Binomial()).fit()
except Exception as e:
    logit_unadj = e

# Adjusted model: gender + continuous numeric controls to avoid collinearity
numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
continuous_controls = [
    c for c in numeric_cols
    if c not in [outcome, gender_col] and analysis_df[c].nunique() > 2
]
X_adj = analysis_df[[gender_col] + continuous_controls]
X_adj = X_adj.loc[:, X_adj.nunique() > 1]
X_adj = sm.add_constant(X_adj, has_constant='add')

# Align and drop rows with missing values for adjusted model
adj_mask = X_adj.notna().all(axis=1) & y.notna()
X_adj_clean = X_adj.loc[adj_mask]
y_adj = y.loc[adj_mask]

logit_adj = None
logit_adj_error = None
try:
    logit_adj = sm.GLM(y_adj, X_adj_clean, family=sm.families.Binomial()).fit()
except Exception as e:
    logit_adj = e
    logit_adj_error = str(e)

def extract_effect(model, var):
    if hasattr(model, 'params') and var in model.params.index:
        coef = model.params[var]
        pval = model.pvalues[var]
        or_val = float(np.exp(coef))
        ci = model.conf_int().loc[var]
        or_ci = [float(np.exp(ci[0])), float(np.exp(ci[1]))]
        return coef, pval, or_val, or_ci
    return None, None, None, None

female_coef, female_p, female_or, female_ci = extract_effect(logit_unadj, gender_col)
female_coef_adj, female_p_adj, female_or_adj, female_ci_adj = extract_effect(logit_adj, gender_col)

# Save results
def to_py(val):
    if isinstance(val, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    return val

results = {
    'n_rows': len(analysis_df),
    'columns': cols,
    'outcome': outcome,
    'outcome_is_denial': outcome_is_denial,
    'accept_deny_complement': complement,
    'gender_column': gender_col,
    'approval_rates_by_female': {to_py(k): to_py(v) for k, v in approval_rates.to_dict().items()},
    'prop_test': {
        'female_approval_rate': to_py(p1),
        'male_approval_rate': to_py(p2),
        'z': to_py(z),
        'p_value': to_py(p_value),
        'n_female': n1,
        'n_male': n2,
    },
    'chi2': {
        'chi2_stat': to_py(chi2_stat),
        'p_value': to_py(chi2_p),
        'dof': chi2[2],
    },
    'logit_female_unadjusted': {
        'coef': to_py(female_coef),
        'p_value': to_py(female_p),
        'odds_ratio': to_py(female_or),
        'odds_ratio_ci': None if female_ci is None else [to_py(female_ci[0]), to_py(female_ci[1])],
    },
    'logit_female_adjusted': {
        'coef': to_py(female_coef_adj),
        'p_value': to_py(female_p_adj),
        'odds_ratio': to_py(female_or_adj),
        'odds_ratio_ci': None if female_ci_adj is None else [to_py(female_ci_adj[0]), to_py(female_ci_adj[1])],
    },
    'logit_female_adjusted_error': logit_adj_error,
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
