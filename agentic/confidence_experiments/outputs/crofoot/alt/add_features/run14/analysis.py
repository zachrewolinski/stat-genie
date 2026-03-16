import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Load data
path = 'crofoot.csv'
df = pd.read_csv(path)

# Keep relevant columns, drop rows with missing values
cols = ['win','n_focal','n_other','dist_focal','dist_other']
sub = df[cols].dropna().copy()

# Derived variables
sub['rel_size'] = sub['n_focal'] - sub['n_other']  # positive => focal larger
sub['rel_dist_100'] = (sub['dist_other'] - sub['dist_focal']) / 100.0  # positive => contest closer to focal

# Logistic regression: win ~ rel_size + rel_dist_100
X = sm.add_constant(sub[['rel_size','rel_dist_100']])
model = sm.Logit(sub['win'], X)
result = model.fit(disp=False)

# Separate models
X_size = sm.add_constant(sub[['rel_size']])
res_size = sm.Logit(sub['win'], X_size).fit(disp=False)

X_dist = sm.add_constant(sub[['rel_dist_100']])
res_dist = sm.Logit(sub['win'], X_dist).fit(disp=False)

# Odds ratios and CIs
params = result.params
conf = result.conf_int()
conf.columns = ['ci_low','ci_high']

or_table = pd.DataFrame({
    'coef': params,
    'or': np.exp(params),
    'ci_low': np.exp(conf['ci_low']),
    'ci_high': np.exp(conf['ci_high']),
    'p': result.pvalues
})

# Point-biserial correlations
pbs_size = stats.pointbiserialr(sub['win'], sub['rel_size'])
pbs_dist = stats.pointbiserialr(sub['win'], sub['rel_dist_100'])

summary = {
    'n': int(sub.shape[0]),
    'mean_win': float(sub['win'].mean()),
    'logit_full': {
        'params': result.params.to_dict(),
        'pvalues': result.pvalues.to_dict(),
        'odds_ratios': or_table['or'].to_dict(),
        'ci_low': or_table['ci_low'].to_dict(),
        'ci_high': or_table['ci_high'].to_dict(),
        'aic': float(result.aic)
    },
    'logit_size_only': {
        'params': res_size.params.to_dict(),
        'pvalues': res_size.pvalues.to_dict(),
        'aic': float(res_size.aic)
    },
    'logit_dist_only': {
        'params': res_dist.params.to_dict(),
        'pvalues': res_dist.pvalues.to_dict(),
        'aic': float(res_dist.aic)
    },
    'pointbiserial': {
        'rel_size_r': float(pbs_size.statistic),
        'rel_size_p': float(pbs_size.pvalue),
        'rel_dist_r': float(pbs_dist.statistic),
        'rel_dist_p': float(pbs_dist.pvalue)
    }
}

print(json.dumps(summary, indent=2))
