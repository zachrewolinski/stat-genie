import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('panda_nuts.csv')

# Basic cleaning
# Ensure categorical types
for col in ['sex', 'help', 'hammer']:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Compute efficiency (nuts per second)
# Avoid division by zero; seconds min seems >0

df['nuts_per_sec'] = df['nuts_opened'] / df['seconds']

df['log_nps'] = np.log1p(df['nuts_per_sec'])

# Summary
summary = {
    'n_rows': len(df),
    'n_chimps': df['chimpanzee'].nunique(),
    'sessions_per_chimp': df['chimpanzee'].value_counts().describe().to_dict(),
    'nuts_per_sec': df['nuts_per_sec'].describe().to_dict(),
}

# OLS on rate
ols = smf.ols('nuts_per_sec ~ age + C(sex) + C(help)', data=df).fit(cov_type='HC3')

# OLS on log1p(rate)
ols_log = smf.ols('log_nps ~ age + C(sex) + C(help)', data=df).fit(cov_type='HC3')

# Poisson GLM with offset (log seconds) to model counts
# nuts_opened can be zero; Poisson OK
# Use log link with offset

df['log_seconds'] = np.log(df['seconds'])
poisson = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=df,
                  family=sm.families.Poisson(), offset=df['log_seconds']).fit(cov_type='HC3')

# Negative binomial GLM as robustness
try:
    nb = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=df,
                 family=sm.families.NegativeBinomial(alpha=1.0), offset=df['log_seconds']).fit(cov_type='HC3')
except Exception as e:
    nb = None
    nb_err = str(e)

# Mixed effects model (random intercept by chimpanzee) on log1p(rate)
try:
    md = smf.mixedlm('log_nps ~ age + C(sex) + C(help)', data=df, groups=df['chimpanzee'])
    mixed = md.fit(method='lbfgs', reml=False)
except Exception as e:
    mixed = None
    mixed_err = str(e)

# Extract key p-values and coefficients

def extract(model, label):
    if model is None:
        return {'label': label, 'error': 'model_failed'}
    params = model.params
    pvals = model.pvalues
    return {
        'label': label,
        'params': params.to_dict(),
        'pvalues': pvals.to_dict(),
        'nobs': int(model.nobs),
        'aic': float(model.aic) if hasattr(model, 'aic') else None,
        'bic': float(model.bic) if hasattr(model, 'bic') else None,
        'rsquared': float(model.rsquared) if hasattr(model, 'rsquared') else None,
    }

results = {
    'summary': summary,
    'ols_rate': extract(ols, 'ols_rate'),
    'ols_log': extract(ols_log, 'ols_log'),
    'poisson': extract(poisson, 'poisson'),
    'neg_binom': extract(nb, 'neg_binom') if nb is not None else {'label': 'neg_binom', 'error': nb_err if 'nb_err' in locals() else 'model_failed'},
    'mixed': extract(mixed, 'mixed') if mixed is not None else {'label': 'mixed', 'error': mixed_err if 'mixed_err' in locals() else 'model_failed'},
}

# Print results in readable form
import json
print(json.dumps(results, indent=2, default=str))
