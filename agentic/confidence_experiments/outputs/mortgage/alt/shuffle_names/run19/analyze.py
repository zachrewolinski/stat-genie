import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
path = "mortgage.csv"
df = pd.read_csv(path)

# Identify candidate gender and outcome columns
# Gender column likely named 'female'
# Outcome columns likely 'deny' or 'accept'

cols = df.columns.tolist()

# Basic stats
summary = {}
for col in ['female', 'deny', 'accept']:
    if col in df.columns:
        summary[col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'unique': int(df[col].nunique())
        }

# Check relationship between deny and accept if both present
rel = None
if 'deny' in df.columns and 'accept' in df.columns:
    rel = {
        'corr': float(df['deny'].corr(df['accept'])),
        'mean_sum': float((df['deny'] + df['accept']).mean()),
        'prop_sum_1': float(((df['deny'] + df['accept']) == 1).mean()),
        'prop_sum_0': float(((df['deny'] + df['accept']) == 0).mean()),
        'prop_sum_2': float(((df['deny'] + df['accept']) == 2).mean()),
    }

# Choose outcome: if deny looks like denial indicator (mean around 0.3) and accept is complement
# We'll decide in analysis step; here compute tests for both.

def analyze_outcome(outcome_col):
    # Ensure binary
    y = df[outcome_col]
    # contingency table female (1) vs male (0)
    ct = pd.crosstab(df['female'], y)
    # If table missing columns 0/1, fill
    for val in [0, 1]:
        if val not in ct.columns:
            ct[val] = 0
    ct = ct[[0, 1]]

    # Rates
    rate_female = ct.loc[1, 1] / ct.loc[1].sum() if 1 in ct.index else np.nan
    rate_male = ct.loc[0, 1] / ct.loc[0].sum() if 0 in ct.index else np.nan
    diff = rate_female - rate_male

    # Chi-square test
    chi2, p, dof, expected = stats.chi2_contingency(ct)

    # Logistic regression (female only)
    X = sm.add_constant(df['female'])
    model = sm.Logit(y, X).fit(disp=False)
    coef = model.params['female']
    pval = model.pvalues['female']
    or_val = float(np.exp(coef))

    return {
        'contingency': ct.to_dict(),
        'rate_female': float(rate_female),
        'rate_male': float(rate_male),
        'rate_diff': float(diff),
        'chi2_p': float(p),
        'logit_coef': float(coef),
        'logit_p': float(pval),
        'odds_ratio': float(or_val),
    }

results = {}
if 'female' in df.columns:
    for outcome in ['deny', 'accept']:
        if outcome in df.columns:
            results[outcome] = analyze_outcome(outcome)

# Add multivariable logistic regression controlling for typical covariates if present
# Choose a set of numeric columns excluding target and female
control_results = {}
for outcome in ['deny', 'accept']:
    if outcome in df.columns and 'female' in df.columns:
        # pick potential controls (numeric) excluding outcome and female
        candidates = [c for c in df.columns if c not in [outcome, 'female']]
        # Use only numeric columns and drop high-cardinality categorical? Keep numeric
        numeric_cols = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
        # To avoid collinearity with other binary like accept/deny, drop other outcome
        if outcome == 'deny' and 'accept' in numeric_cols:
            numeric_cols.remove('accept')
        if outcome == 'accept' and 'deny' in numeric_cols:
            numeric_cols.remove('deny')
        # Limit to a reasonable set to avoid issues: use all numeric cols but drop any with too many missing
        valid_cols = []
        for c in numeric_cols:
            if df[c].isna().mean() < 0.05:
                valid_cols.append(c)
        # Prepare design matrix
        X = df[['female'] + valid_cols].copy()
        # Drop rows with missing
        data = pd.concat([df[outcome], X], axis=1).dropna()
        y = data[outcome]
        X = data.drop(columns=[outcome])
        X = sm.add_constant(X)
        # Fit logistic regression if possible
        try:
            model = sm.Logit(y, X).fit(disp=False)
            coef = model.params.get('female', np.nan)
            pval = model.pvalues.get('female', np.nan)
            or_val = float(np.exp(coef)) if pd.notna(coef) else np.nan
            control_results[outcome] = {
                'n': int(model.nobs),
                'female_coef': float(coef),
                'female_p': float(pval),
                'female_or': float(or_val),
                'num_controls': len(valid_cols),
                'controls': valid_cols,
            }
        except Exception as e:
            control_results[outcome] = {'error': str(e), 'num_controls': len(valid_cols), 'controls': valid_cols}

output = {
    'summary': summary,
    'deny_accept_relation': rel,
    'results': results,
    'control_results': control_results,
    'n_rows': int(len(df)),
    'columns': cols,
}

print(json.dumps(output, indent=2))
