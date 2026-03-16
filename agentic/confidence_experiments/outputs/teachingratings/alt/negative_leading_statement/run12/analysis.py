import json
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

# Load data
_df = pd.read_csv('teachingratings.csv')

# Basic cleaning: drop rows with missing values in key columns
key_cols = ['beauty', 'eval']
_df = _df.dropna(subset=key_cols)

# Correlations
pearson_r, pearson_p = stats.pearsonr(_df['beauty'], _df['eval'])
spearman_r, spearman_p = stats.spearmanr(_df['beauty'], _df['eval'])

# Simple OLS
model_simple = smf.ols('eval ~ beauty', data=_df).fit()

# Multivariate OLS with controls
# Treat categorical variables as categorical with C()
formula = (
    'eval ~ beauty + age + students + allstudents '
    '+ C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits)'
)
model_full = smf.ols(formula, data=_df).fit()

summary = {
    'n': int(_df.shape[0]),
    'pearson_r': pearson_r,
    'pearson_p': pearson_p,
    'spearman_r': spearman_r,
    'spearman_p': spearman_p,
    'simple_coef': model_simple.params['beauty'],
    'simple_p': model_simple.pvalues['beauty'],
    'simple_r2': model_simple.rsquared,
    'full_coef': model_full.params['beauty'],
    'full_p': model_full.pvalues['beauty'],
    'full_r2': model_full.rsquared,
}

print(json.dumps(summary, indent=2))
