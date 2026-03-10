import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('panda_nuts.csv')

# Clean / derive variables
_df['sex'] = _df['sex'].astype('category')
_df['help'] = _df['help'].astype('category')
_df['efficiency'] = _df['nuts_opened'] / _df['seconds']  # nuts per second

# Summary stats
summary = {
    'n_rows': int(_df.shape[0]),
    'n_chimps': int(_df['chimpanzee'].nunique()),
    'efficiency_mean': float(_df['efficiency'].mean()),
    'efficiency_median': float(_df['efficiency'].median()),
}

# Correlation between age and efficiency
summary['age_efficiency_corr'] = float(_df['age'].corr(_df['efficiency']))

# Group means
summary['efficiency_by_sex'] = _df.groupby('sex')['efficiency'].mean().to_dict()
summary['efficiency_by_help'] = _df.groupby('help')['efficiency'].mean().to_dict()

# GLM Poisson with offset (log seconds)
# nuts_opened ~ age + sex + help + offset(log(seconds))
_df['log_seconds'] = np.log(_df['seconds'])

# Poisson
poisson_model = smf.glm(
    formula='nuts_opened ~ age + sex + help',
    data=_df,
    family=sm.families.Poisson(),
    offset=_df['log_seconds'],
).fit(cov_type='cluster', cov_kwds={'groups': _df['chimpanzee']})

# Negative Binomial (to handle overdispersion)
nb_model = smf.glm(
    formula='nuts_opened ~ age + sex + help',
    data=_df,
    family=sm.families.NegativeBinomial(alpha=1.0),
    offset=_df['log_seconds'],
).fit(cov_type='cluster', cov_kwds={'groups': _df['chimpanzee']})

# Linear model on efficiency (robust SE, cluster by chimp)
lin_model = smf.ols(
    formula='efficiency ~ age + sex + help',
    data=_df,
).fit(cov_type='cluster', cov_kwds={'groups': _df['chimpanzee']})


def _extract(model):
    return {
        'params': model.params.to_dict(),
        'pvalues': model.pvalues.to_dict(),
        'bse': model.bse.to_dict(),
        'nobs': int(model.nobs),
    }

results = {
    'summary': summary,
    'poisson': _extract(poisson_model),
    'neg_bin': _extract(nb_model),
    'linear_efficiency': _extract(lin_model),
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)
