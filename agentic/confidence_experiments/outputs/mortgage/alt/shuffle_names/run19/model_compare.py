import pandas as pd
import numpy as np
import statsmodels.api as sm
import json


df = pd.read_csv('mortgage.csv')

gender_col = 'denied_PMI'  # per info.json description
accept_col = 'deny'        # complement of self_employed

def fit_logit(cols):
    data = df[[accept_col] + cols].dropna()
    y = data[accept_col]
    X = sm.add_constant(data[cols])
    try:
        model = sm.Logit(y, X).fit(disp=False)
        coef = model.params.get(gender_col, np.nan)
        pval = model.pvalues.get(gender_col, np.nan)
        or_val = float(np.exp(coef)) if pd.notna(coef) else np.nan
        return {
            'n': int(model.nobs),
            'coef': float(coef),
            'p': float(pval),
            'or': float(or_val),
        }
    except Exception as e:
        return {'error': str(e)}

# Identify continuous vs binary
binary_cols = [c for c in df.columns if df[c].dropna().nunique() == 2]
continuous_cols = [c for c in df.columns if c not in binary_cols]

# Remove outcome columns and gender from controls
binary_controls = [c for c in binary_cols if c not in [accept_col, gender_col, 'self_employed']]
continuous_controls = [c for c in continuous_cols if c not in [accept_col, gender_col]]

results = {}
results['unadjusted'] = fit_logit([gender_col])
results['continuous_only'] = fit_logit([gender_col] + continuous_controls)
results['binary_only'] = fit_logit([gender_col] + binary_controls)
results['all_controls'] = fit_logit([gender_col] + continuous_controls + binary_controls)

output = {
    'gender_col': gender_col,
    'accept_col': accept_col,
    'binary_controls': binary_controls,
    'continuous_controls': continuous_controls,
    'results': results,
}

print(json.dumps(output, indent=2))
