import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.sandwich_covariance import cov_cluster

# Load data
_df = pd.read_csv('panda_nuts.csv')

# Basic cleaning / types
_df['sex'] = _df['sex'].astype('category')
_df['help'] = _df['help'].astype('category')

# Poisson GLM with exposure (seconds)
poisson_model = smf.glm(
    'nuts_opened ~ age + C(sex) + C(help)',
    data=_df,
    family=sm.families.Poisson(),
    offset=np.log(_df['seconds'])
)
poisson_res = poisson_model.fit(cov_type='cluster', cov_kwds={'groups': _df['chimpanzee']})

# Overdispersion diagnostic
pearson_chi2 = np.sum(poisson_res.resid_pearson ** 2)
poisson_dispersion = pearson_chi2 / poisson_res.df_resid if poisson_res.df_resid > 0 else np.nan

# Negative Binomial (NB2) with offset, cluster-robust SE
X = pd.get_dummies(_df[['age', 'sex', 'help']], drop_first=True)
X = sm.add_constant(X)
y = _df['nuts_opened']
offset = np.log(_df['seconds'])

nb_model = sm.NegativeBinomial(y, X, offset=offset)
nb_res = nb_model.fit(disp=0)
# statsmodels NegativeBinomialResults lacks public get_robustcov_results in this version
# Cluster-robust covariance for NB results
nb_cov = cov_cluster(nb_res, _df['chimpanzee'])
nb_params = nb_res.params
nb_se = pd.Series(np.sqrt(np.diag(nb_cov)), index=nb_params.index)
nb_z = nb_params / nb_se
nb_p = pd.Series(2 * (1 - stats.norm.cdf(np.abs(nb_z))), index=nb_params.index)

# Helper to extract rate ratios and CI

def _rr_ci(res, term):
    coef = res.params[term]
    se = res.bse[term]
    rr = float(np.exp(coef))
    ci_low = float(np.exp(coef - 1.96 * se))
    ci_high = float(np.exp(coef + 1.96 * se))
    p = float(res.pvalues[term])
    return {'rr': rr, 'ci_low': ci_low, 'ci_high': ci_high, 'p': p}


def _rr_ci_nb(term):
    coef = nb_params[term]
    se = nb_se[term]
    rr = float(np.exp(coef))
    ci_low = float(np.exp(coef - 1.96 * se))
    ci_high = float(np.exp(coef + 1.96 * se))
    p = float(nb_p[term])
    return {'rr': rr, 'ci_low': ci_low, 'ci_high': ci_high, 'p': p}

# Terms for each model
poisson_terms = {
    'age': 'age',
    'sex_m': 'C(sex)[T.m]',
    'help_y': 'C(help)[T.y]'
}
nb_terms = {
    'age': 'age',
    'sex_m': 'sex_m',
    'help_y': 'help_y'
}

poisson_effects = {k: _rr_ci(poisson_res, v) for k, v in poisson_terms.items()}
nb_effects = {k: _rr_ci_nb(v) for k, v in nb_terms.items()}

summary = {
    'n_rows': int(len(_df)),
    'n_chimpanzees': int(_df['chimpanzee'].nunique()),
    'poisson_dispersion': float(poisson_dispersion),
    'poisson_effects': poisson_effects,
    'nb_alpha': float(nb_res.params['alpha']) if 'alpha' in nb_res.params else None,
    'nb_effects': nb_effects,
}

with open('analysis_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
