import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = 'hurricane.csv'

df = pd.read_csv(DATA_PATH)

# Basic checks
summary = {
    'rows': len(df),
    'columns': list(df.columns),
    'missing_by_col': df.isna().sum().to_dict(),
}

# Define key variables
# feature4: masculinity-femininity index (higher=more feminine)
# feature6: binary female (1) vs male (0) name
# feature8: deaths (count)
# feature7: category
# feature5: min pressure
# feature13: max wind speed

# Correlations (Pearson and Spearman) between femininity index and deaths
fem = df['feature4']
deaths = df['feature8']

pearson = stats.pearsonr(fem, deaths)
spearman = stats.spearmanr(fem, deaths)

# Log-transform deaths for regression stability (add 1 for zeros)
df['log_deaths'] = np.log1p(df['feature8'])

# Baseline regression: log deaths ~ femininity index + category + min pressure + max wind speed
# Also include year to control for time trends (feature2) and elapsed years (feature10) is redundant with year.
model_formula = 'log_deaths ~ feature4 + feature7 + feature5 + feature13 + feature2'
model = smf.ols(model_formula, data=df).fit()

# Alternative: using binary female indicator
model_formula_bin = 'log_deaths ~ feature6 + feature7 + feature5 + feature13 + feature2'
model_bin = smf.ols(model_formula_bin, data=df).fit()

# Robustness: Poisson regression on deaths (count) with same covariates
poisson = smf.glm('feature8 ~ feature4 + feature7 + feature5 + feature13 + feature2',
                  data=df, family=sm.families.Poisson()).fit()

output = {
    'summary': summary,
    'pearson_r': pearson[0],
    'pearson_p': pearson[1],
    'spearman_r': spearman.correlation,
    'spearman_p': spearman.pvalue,
    'ols_fem': {
        'coef': model.params.get('feature4'),
        'pval': model.pvalues.get('feature4'),
        'conf_int': model.conf_int().loc['feature4'].tolist(),
        'r2': model.rsquared,
    },
    'ols_bin': {
        'coef': model_bin.params.get('feature6'),
        'pval': model_bin.pvalues.get('feature6'),
        'conf_int': model_bin.conf_int().loc['feature6'].tolist(),
        'r2': model_bin.rsquared,
    },
    'poisson_fem': {
        'coef': poisson.params.get('feature4'),
        'pval': poisson.pvalues.get('feature4'),
        'conf_int': poisson.conf_int().loc['feature4'].tolist(),
    },
}

print(json.dumps(output, indent=2))
