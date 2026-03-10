import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Map shuffled columns based on metadata descriptions
# age (years) is in column 'hammer' per metadata description
# sex is in column 'nuts_opened' (m/f)
# help (received help) is in column 'seconds' (y/N)
# nuts opened is in column 'help'
# session duration seconds is in column 'chimpanzee'

df = df.rename(columns={
    'hammer': 'age_years',
    'nuts_opened': 'sex',
    'seconds': 'helped',
    'help': 'nuts_opened',
    'chimpanzee': 'duration_s',
    'sex': 'hammer_type'
})

# Clean / encode
# Helped: map y/N to yes/no
help_map = {'y': 'yes', 'Y': 'yes', 'N': 'no', 'n': 'no'}
df['helped'] = df['helped'].map(help_map)

# Efficiency: nuts opened per second
# Avoid division by zero; duration min is 2.5 in data

df['efficiency'] = df['nuts_opened'] / df['duration_s']
# Log1p transform for robustness
df['log_efficiency'] = np.log1p(df['efficiency'])

# Basic summaries
def to_native(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x

summary = {
    'n': int(len(df)),
    'efficiency_mean': float(df['efficiency'].mean()),
    'efficiency_median': float(df['efficiency'].median()),
    'age_mean': float(df['age_years'].mean()),
    'age_min': float(df['age_years'].min()),
    'age_max': float(df['age_years'].max()),
    'sex_counts': {k: int(v) for k, v in df['sex'].value_counts().to_dict().items()},
    'helped_counts': {k: int(v) for k, v in df['helped'].value_counts().to_dict().items()},
}

# Group means
summary['efficiency_by_sex'] = {k: float(v) for k, v in df.groupby('sex')['efficiency'].mean().to_dict().items()}
summary['efficiency_by_helped'] = {k: float(v) for k, v in df.groupby('helped')['efficiency'].mean().to_dict().items()}

# OLS on log efficiency with robust SE
model_log = smf.ols('log_efficiency ~ age_years + C(sex) + C(helped)', data=df).fit(cov_type='HC3')

# OLS on raw efficiency with robust SE
model_raw = smf.ols('efficiency ~ age_years + C(sex) + C(helped)', data=df).fit(cov_type='HC3')

# Poisson GLM on counts with offset for duration
import statsmodels.api as sm

df['log_duration'] = np.log(df['duration_s'])
model_glm = smf.glm('nuts_opened ~ age_years + C(sex) + C(helped)', data=df,
                    family=sm.families.Poisson(), offset=df['log_duration']).fit(cov_type='HC3')

# Collect key results

def extract_terms(model, terms):
    out = {}
    for term in terms:
        if term in model.params.index:
            out[term] = {
                'coef': float(model.params[term]),
                'pvalue': float(model.pvalues[term]),
                'ci_low': float(model.conf_int().loc[term, 0]),
                'ci_high': float(model.conf_int().loc[term, 1]),
            }
    return out

terms = ['age_years', 'C(sex)[T.m]', 'C(helped)[T.yes]']

results = {
    'summary': summary,
    'model_log': {
        'r2': float(model_log.rsquared),
        'n': int(model_log.nobs),
        'terms': extract_terms(model_log, terms),
    },
    'model_raw': {
        'r2': float(model_raw.rsquared),
        'n': int(model_raw.nobs),
        'terms': extract_terms(model_raw, terms),
    },
    'model_glm': {
        'n': int(model_glm.nobs),
        'terms': extract_terms(model_glm, terms),
        'aic': float(model_glm.aic),
    }
}

print(json.dumps(results, indent=2))
