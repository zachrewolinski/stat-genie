import pandas as pd
import statsmodels.formula.api as smf
import json

# Load data
_df = pd.read_csv('teachingratings.csv')

# Ensure categorical columns are treated as category
cat_cols = ['eval', 'tenure', 'prof', 'native', 'gender', 'credits']
for c in cat_cols:
    _df[c] = _df[c].astype('category')

# Outcome and predictor
# Simple model: evaluation score vs beauty
model_simple = smf.ols('allstudents ~ beauty', data=_df).fit()

# Adjusted model with covariates suggested by metadata
# Note: avoid including 'division' (unique per row) to prevent perfect fit.
# Treat instructor id (students) as categorical? We'll keep it numeric as in dataset to avoid overfitting.
model_adj = smf.ols(
    'allstudents ~ beauty + age + C(tenure) + C(prof) + C(native) + C(gender) + C(credits) + C(eval) + rownames + minority',
    data=_df,
).fit()

# Extract key stats
summary = {
    'n': int(_df.shape[0]),
    'simple_coef': model_simple.params['beauty'],
    'simple_p': model_simple.pvalues['beauty'],
    'simple_ci': model_simple.conf_int().loc['beauty'].tolist(),
    'adj_coef': model_adj.params['beauty'],
    'adj_p': model_adj.pvalues['beauty'],
    'adj_ci': model_adj.conf_int().loc['beauty'].tolist(),
    'simple_r2': model_simple.rsquared,
    'adj_r2': model_adj.rsquared,
}

print(json.dumps(summary, indent=2))
