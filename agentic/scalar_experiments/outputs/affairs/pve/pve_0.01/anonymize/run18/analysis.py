import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('affairs.csv')

# Ensure expected columns exist
required = {'feature2','feature6','feature3','feature4','feature5','feature7','feature8','feature9','feature10'}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Clean feature6
children = df['feature6'].astype(str).str.strip().str.lower()
if not set(children.unique()).issubset({'yes','no'}):
    # try map common variants
    children = children.replace({'y':'yes','n':'no','1':'yes','0':'no'})

# Numeric outcome
y = pd.to_numeric(df['feature2'], errors='coerce')

# Group stats
mask_yes = children == 'yes'
mask_no = children == 'no'

# Drop rows with missing
valid = y.notna() & (mask_yes | mask_no)

y = y[valid]
mask_yes = mask_yes[valid]
mask_no = mask_no[valid]

stats_out = {}
for label, m in [('yes', mask_yes), ('no', mask_no)]:
    vals = y[m]
    stats_out[label] = {
        'n': int(vals.shape[0]),
        'mean': float(vals.mean()),
        'median': float(vals.median()),
        'std': float(vals.std(ddof=1)),
        'prop_zero': float((vals == 0).mean()),
    }

# T-test (Welch)

t_res = stats.ttest_ind(y[mask_yes], y[mask_no], equal_var=False, nan_policy='omit')

# Mann-Whitney U (two-sided)
try:
    mw_res = stats.mannwhitneyu(y[mask_yes], y[mask_no], alternative='two-sided')
    mw_stat = float(mw_res.statistic)
    mw_p = float(mw_res.pvalue)
except Exception:
    mw_stat = float('nan')
    mw_p = float('nan')

# OLS with controls
# Use categorical for gender and children

df_model = df.copy()
df_model['children'] = children
# Drop missing
model_cols = ['feature2','children','feature3','feature4','feature5','feature7','feature8','feature9','feature10']
df_model = df_model[model_cols].dropna()

ols = smf.ols('feature2 ~ C(children) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10', data=df_model).fit()

coef_children = ols.params.get('C(children)[T.yes]', np.nan)
p_children = ols.pvalues.get('C(children)[T.yes]', np.nan)

# Logistic regression: any affairs

df_model['any_affair'] = (df_model['feature2'] > 0).astype(int)
logit = smf.logit('any_affair ~ C(children) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10', data=df_model).fit(disp=False)

logit_coef = logit.params.get('C(children)[T.yes]', np.nan)
logit_p = logit.pvalues.get('C(children)[T.yes]', np.nan)
logit_or = float(np.exp(logit_coef)) if pd.notna(logit_coef) else float('nan')

results = {
    'group_stats': stats_out,
    'ttest': {'statistic': float(t_res.statistic), 'pvalue': float(t_res.pvalue)},
    'mannwhitneyu': {'statistic': mw_stat, 'pvalue': mw_p},
    'ols_children_coef': float(coef_children),
    'ols_children_pvalue': float(p_children),
    'logit_children_coef': float(logit_coef),
    'logit_children_pvalue': float(logit_p),
    'logit_children_or': float(logit_or),
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
