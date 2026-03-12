import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

path = 'crofoot.csv'
df = pd.read_csv(path)

# outcome
outcome = df['feature4']

# relative group size (focal - other)
rel_size = df['feature7'] - df['feature8']

# location advantage: positive if contest closer to focal group's center
# (other distance - focal distance)
loc_adv = df['feature6'] - df['feature5']

# Assemble dataframe
X = pd.DataFrame({
    'rel_size': rel_size,
    'loc_adv': loc_adv,
})

# Standardize predictors for stable estimation and comparable effect sizes
X_std = (X - X.mean()) / X.std(ddof=0)
X_std = sm.add_constant(X_std)

model = sm.Logit(outcome, X_std)
result = model.fit(disp=False)

print(result.summary())

# Compute odds ratios for 1 SD change
params = result.params
conf = result.conf_int()

odds_ratios = np.exp(params)
conf_or = np.exp(conf)

print("\nOdds ratios (per 1 SD):")
for name in odds_ratios.index:
    print(name, odds_ratios[name], conf_or.loc[name].tolist())

# Likelihood ratio test against intercept-only
null_model = sm.Logit(outcome, sm.add_constant(pd.DataFrame({'const': np.ones(len(outcome))}))).fit(disp=False)
llr = 2 * (result.llf - null_model.llf)
df_diff = result.df_model
p_llr = stats.chi2.sf(llr, df_diff)
print("\nLLR test:", llr, df_diff, p_llr)

# Simple correlations for context
print("\nCorrelations: outcome vs predictors")
print("rel_size", np.corrcoef(outcome, rel_size)[0,1])
print("loc_adv", np.corrcoef(outcome, loc_adv)[0,1])

# Save key stats to json for convenience
import json
out = {
    'n': len(df),
    'coef': result.params.to_dict(),
    'pvalues': result.pvalues.to_dict(),
    'odds_ratios': odds_ratios.to_dict(),
    'or_conf_int': {k: conf_or.loc[k].tolist() for k in conf_or.index},
    'llr_pvalue': p_llr,
}
with open('analysis_results.json', 'w') as f:
    json.dump(out, f, indent=2)
