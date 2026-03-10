import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
path = 'crofoot.csv'
df = pd.read_csv(path)

# Create predictors
# Relative group size: focal size minus other size
# Location advantage: other distance from its center minus focal distance from its center
# positive -> contest closer to focal home range center

df['rel_size'] = df['n_focal'] - df['n_other']
df['loc_adv'] = df['dist_other'] - df['dist_focal']

# Basic summaries
summary = {
    'n': len(df),
    'win_rate': df['win'].mean(),
    'rel_size_mean': df['rel_size'].mean(),
    'loc_adv_mean': df['loc_adv'].mean(),
}

# Logistic regression
X = sm.add_constant(df[['rel_size', 'loc_adv']])
y = df['win']

model = sm.Logit(y, X)
result = model.fit(disp=False)

# Compute odds ratios and CI
params = result.params
conf = result.conf_int()
conf.columns = ['ci_low', 'ci_high']

odds = np.exp(params)
ci_low = np.exp(conf['ci_low'])
ci_high = np.exp(conf['ci_high'])

# Package results
output = {
    'summary': summary,
    'logit_params': params.to_dict(),
    'logit_pvalues': result.pvalues.to_dict(),
    'odds_ratios': odds.to_dict(),
    'odds_ci_low': ci_low.to_dict(),
    'odds_ci_high': ci_high.to_dict(),
    'pseudo_r2': result.prsquared,
}

# Standardized predictors for effect size comparison
df_std = df.copy()
df_std[['rel_size', 'loc_adv']] = (df_std[['rel_size', 'loc_adv']] - df_std[['rel_size', 'loc_adv']].mean()) / df_std[['rel_size', 'loc_adv']].std(ddof=0)
X_std = sm.add_constant(df_std[['rel_size', 'loc_adv']])
model_std = sm.Logit(y, X_std)
result_std = model_std.fit(disp=False)
output['logit_std_params'] = result_std.params.to_dict()
output['logit_std_pvalues'] = result_std.pvalues.to_dict()

# Group comparisons: win vs loss
win = df[df['win'] == 1]
loss = df[df['win'] == 0]
t_rel = stats.ttest_ind(win['rel_size'], loss['rel_size'], equal_var=False)
t_loc = stats.ttest_ind(win['loc_adv'], loss['loc_adv'], equal_var=False)
output['ttest_rel_size'] = {'t': float(t_rel.statistic), 'p': float(t_rel.pvalue)}
output['ttest_loc_adv'] = {'t': float(t_loc.statistic), 'p': float(t_loc.pvalue)}

print(output)
