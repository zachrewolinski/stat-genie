import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats


df = pd.read_csv('hurricane.csv')

# Transformations
for col in ['feature8', 'feature9', 'feature14']:
    if col in df.columns:
        df[f'log_{col}'] = np.log1p(df[col])

df['log_deaths'] = np.log1p(df['feature8'])

# Helper to extract coef/pval/ci

def extract(model, term):
    return {
        'coef': model.params.get(term),
        'pval': model.pvalues.get(term),
        'conf_int': model.conf_int().loc[term].tolist()
    }

# OLS with feature4 and feature12
ols_f4 = smf.ols('log_deaths ~ feature4 + feature7 + feature5 + feature13 + feature2', data=df).fit()
ols_f12 = smf.ols('log_deaths ~ feature12 + feature7 + feature5 + feature13 + feature2', data=df).fit()

# Robust regression (Huber) for feature4
rlm_f4 = smf.rlm('log_deaths ~ feature4 + feature7 + feature5 + feature13 + feature2', data=df).fit()

# Negative binomial for feature4
nb_f4 = smf.glm('feature8 ~ feature4 + feature7 + feature5 + feature13 + feature2',
                data=df, family=sm.families.NegativeBinomial()).fit()

# Simple group comparison (binary gender)
group_stats = df.groupby('feature6')['log_deaths'].agg(['mean', 'median', 'count']).to_dict()

# Correlations for feature12 as alternative femininity rating
pearson_f12 = stats.pearsonr(df['feature12'], df['feature8'])
spearman_f12 = stats.spearmanr(df['feature12'], df['feature8'])

out = {
    'ols_feature4': extract(ols_f4, 'feature4') | {'r2': ols_f4.rsquared},
    'ols_feature12': extract(ols_f12, 'feature12') | {'r2': ols_f12.rsquared},
    'rlm_feature4': {'coef': rlm_f4.params.get('feature4')},
    'nb_feature4': extract(nb_f4, 'feature4'),
    'group_log_deaths_by_gender': group_stats,
    'pearson_feature12': {'r': pearson_f12[0], 'p': pearson_f12[1]},
    'spearman_feature12': {'r': spearman_f12.correlation, 'p': spearman_f12.pvalue}
}

print(json.dumps(out, indent=2))
