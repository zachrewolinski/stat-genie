import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load data

df = pd.read_csv('mortgage.csv')

# Load metadata to identify gender column
with open('info.json', 'r') as f:
    info = json.load(f)

fields = info.get('data_desc', {}).get('fields', [])

gender_col = None
for field in fields:
    desc = (field.get('properties', {}) or {}).get('description', '')
    col = field.get('column')
    if desc and 'female' in desc.lower() and 'male' in desc.lower():
        if col in df.columns:
            gender_col = col
            break

# Fallback if not found
if gender_col is None:
    if 'female' in df.columns:
        gender_col = 'female'
    else:
        raise ValueError('Could not identify gender column.')

# Identify approval outcome
approval_col = None
complement_ok = False

# Prefer the complementary pair self_employed + deny (exactly 1)
if 'self_employed' in df.columns and 'deny' in df.columns:
    se = df['self_employed']
    dy = df['deny']
    if se.dropna().isin([0, 1]).all() and dy.dropna().isin([0, 1]).all():
        if np.all((se + dy).round(6) == 1):
            approval_col = 'deny'
            complement_ok = True

# Fallback to accept/deny complement
if approval_col is None and 'accept' in df.columns and 'deny' in df.columns:
    accept = df['accept']
    deny = df['deny']
    if accept.dropna().isin([0, 1]).all() and deny.dropna().isin([0, 1]).all():
        if np.all((accept + deny).round(6) == 1):
            approval_col = 'accept'
            complement_ok = True

# Last resort heuristic
if approval_col is None:
    if 'deny' in df.columns:
        approval_col = 'deny'
    elif 'accept' in df.columns:
        approval_col = 'accept'

if approval_col is None:
    raise ValueError('Could not infer approval outcome from columns.')

# Define approval
analysis_df = df[[gender_col, approval_col]].dropna().copy()
analysis_df = analysis_df.rename(columns={approval_col: 'approval', gender_col: 'gender'})

# Crosstab and rates
ct = pd.crosstab(analysis_df['gender'], analysis_df['approval'])

# Rates by gender
rate_by_gender = analysis_df.groupby('gender')['approval'].mean()

# Chi-square test of independence
chi2, p_chi, dof, expected = stats.chi2_contingency(ct)

# Unadjusted logistic regression: approval ~ gender
X_unadj = sm.add_constant(analysis_df['gender'])
model_unadj = sm.Logit(analysis_df['approval'], X_unadj).fit(disp=0)

# Adjusted logistic regression with all other numeric predictors
exclude_cols = {'approval', approval_col, 'accept', 'deny', 'female', gender_col}
feature_cols = [c for c in df.columns if c not in exclude_cols]
feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

adj_df = df[[approval_col, gender_col] + feature_cols].copy()
adj_df = adj_df.dropna()
adj_df = adj_df.rename(columns={approval_col: 'approval', gender_col: 'gender'})

# Remove zero-variance columns
zero_var_cols = [c for c in feature_cols if adj_df[c].nunique(dropna=True) <= 1]
feature_cols = [c for c in feature_cols if c not in zero_var_cols]

X_features = adj_df[['gender'] + feature_cols].copy()

# Reduce multicollinearity via VIF (keep gender)

def reduce_vif(X, thresh=10.0, protected_cols=None):
    if protected_cols is None:
        protected_cols = set()
    X = X.copy()
    while X.shape[1] > 2:
        vif = pd.Series(
            [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
            index=X.columns,
        )
        max_vif = vif.max()
        if np.isfinite(max_vif) and max_vif <= thresh:
            break
        drop_candidates = vif.sort_values(ascending=False).index.tolist()
        drop_col = None
        for col in drop_candidates:
            if col not in protected_cols:
                drop_col = col
                break
        if drop_col is None:
            break
        X = X.drop(columns=[drop_col])
    return X

X_reduced = reduce_vif(X_features, thresh=10.0, protected_cols={'gender'})

# Fit adjusted model with reduced features
X_adj = sm.add_constant(X_reduced)

model_adj = None
adj_fit_method = None
try:
    model_adj = sm.Logit(adj_df['approval'], X_adj).fit(disp=0, maxiter=200)
    adj_fit_method = 'logit'
except Exception:
    try:
        model_adj = sm.GLM(adj_df['approval'], X_adj, family=sm.families.Binomial()).fit()
        adj_fit_method = 'glm_binomial'
    except Exception:
        model_adj = sm.Logit(adj_df['approval'], X_adj).fit_regularized(disp=0)
        adj_fit_method = 'logit_regularized'

# Extract gender effect

def odds_ratio_and_ci(model, var='gender'):
    coef = model.params[var]
    if hasattr(model, 'bse') and var in model.bse:
        se = model.bse[var]
        or_val = float(np.exp(coef))
        ci_low = float(np.exp(coef - 1.96 * se))
        ci_high = float(np.exp(coef + 1.96 * se))
        p_val = float(model.pvalues[var]) if hasattr(model, 'pvalues') else float('nan')
    else:
        or_val = float(np.exp(coef))
        ci_low, ci_high, p_val = float('nan'), float('nan'), float('nan')
    return or_val, ci_low, ci_high, p_val

or_unadj, ci_low_unadj, ci_high_unadj, p_unadj = odds_ratio_and_ci(model_unadj)
or_adj, ci_low_adj, ci_high_adj, p_adj = odds_ratio_and_ci(model_adj)

results = {
    'n_total': int(len(df)),
    'n_analysis': int(len(analysis_df)),
    'gender_col_used': gender_col,
    'approval_col_used': approval_col,
    'complement_ok': bool(complement_ok),
    'approval_rate_gender_0': float(rate_by_gender.get(0, np.nan)),
    'approval_rate_gender_1': float(rate_by_gender.get(1, np.nan)),
    'rate_diff_gender1_minus_0': float(rate_by_gender.get(1, np.nan) - rate_by_gender.get(0, np.nan)),
    'chi2_p': float(p_chi),
    'or_unadj': or_unadj,
    'ci_unadj': [ci_low_unadj, ci_high_unadj],
    'p_unadj': p_unadj,
    'or_adj': or_adj,
    'ci_adj': [ci_low_adj, ci_high_adj],
    'p_adj': p_adj,
    'feature_cols_initial': feature_cols,
    'feature_cols_final': [c for c in X_reduced.columns if c != 'gender'],
    'adjusted_fit_method': adj_fit_method,
    'zero_var_cols_dropped': zero_var_cols,
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
