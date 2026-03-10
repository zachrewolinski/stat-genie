import json
import pandas as pd
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('teachingratings.csv')

# Basic cleanup: ensure categorical columns are treated as categories
cat_cols = ['minority', 'gender', 'credits', 'division', 'native', 'tenure']
for c in cat_cols:
    if c in _df.columns:
        _df[c] = _df[c].astype('category')

# Simple correlation
corr = _df[['beauty', 'eval']].corr().iloc[0,1]

# Simple OLS
model_simple = smf.ols('eval ~ beauty', data=_df).fit()

# Multiple OLS with controls
# Keep typical controls from dataset
controls = ['age', 'gender', 'minority', 'credits', 'division', 'native', 'tenure', 'students', 'allstudents']
# Build formula
formula = 'eval ~ beauty'
for c in controls:
    if c in cat_cols:
        formula += f' + C({c})'
    else:
        formula += f' + {c}'

model_controls = smf.ols(formula, data=_df).fit()

results = {
    'n': int(_df.shape[0]),
    'corr_beauty_eval': float(corr),
    'simple_coef': float(model_simple.params['beauty']),
    'simple_pvalue': float(model_simple.pvalues['beauty']),
    'simple_ci': [float(x) for x in model_simple.conf_int().loc['beauty']],
    'simple_r2': float(model_simple.rsquared),
    'controls_formula': formula,
    'controls_coef': float(model_controls.params['beauty']),
    'controls_pvalue': float(model_controls.pvalues['beauty']),
    'controls_ci': [float(x) for x in model_controls.conf_int().loc['beauty']],
    'controls_r2': float(model_controls.rsquared)
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
