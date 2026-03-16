import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint

# Load data
csv_path = 'crofoot.csv'
df = pd.read_csv(csv_path)

# Derived variables
# Relative group size (focal minus other)
df['rel_size'] = df['n_focal'] - df['n_other']
# Relative location advantage: positive means contest closer to focal group center
# (other distance larger than focal distance)
df['loc_adv'] = df['dist_other'] - df['dist_focal']

# Basic summaries
n = len(df)
win_rate = df['win'].mean()

# Logistic regression: win ~ rel_size + loc_adv
X = df[['rel_size', 'loc_adv']]
X = sm.add_constant(X)
model = sm.Logit(df['win'], X)
try:
    res = model.fit(disp=False)
except Exception:
    # Fallback to GLM binomial if separation issues
    res = sm.GLM(df['win'], X, family=sm.families.Binomial()).fit()

# Odds ratios and p-values
params = res.params
pvalues = res.pvalues
conf = res.conf_int()

odds_ratios = np.exp(params)
conf_or = np.exp(conf)

# Compare model to intercept-only
X0 = sm.add_constant(pd.DataFrame({'intercept_only': np.ones(n)}))
model0 = sm.Logit(df['win'], X0)
try:
    res0 = model0.fit(disp=False)
except Exception:
    res0 = sm.GLM(df['win'], X0, family=sm.families.Binomial()).fit()

# Likelihood ratio test
lr_stat = 2 * (res.llf - res0.llf)
# df difference = 2
from scipy.stats import chi2
lr_p = chi2.sf(lr_stat, df=2)

# Simple nonparametric comparisons for sanity
# Split by rel_size >0, =0, <0
cats = pd.cut(df['rel_size'], bins=[-np.inf, -0.1, 0.1, np.inf], labels=['smaller','equal','larger'])
win_by_size = df.groupby(cats)['win'].mean().to_dict()

# Split by loc_adv >0 (closer to focal) vs <0
loc_cat = np.where(df['loc_adv'] > 0, 'closer_to_focal', 'closer_to_other_or_equal')
win_by_loc = df.groupby(loc_cat)['win'].mean().to_dict()

# Write results to a json file for the assistant to read
out = {
    'n': n,
    'win_rate': win_rate,
    'params': params.to_dict(),
    'pvalues': pvalues.to_dict(),
    'odds_ratios': odds_ratios.to_dict(),
    'conf_or': conf_or.rename(columns={0:'low',1:'high'}).to_dict(orient='index'),
    'lr_stat': float(lr_stat),
    'lr_p': float(lr_p),
    'win_by_size': win_by_size,
    'win_by_loc': win_by_loc,
}

with open('analysis_results.json', 'w') as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
