import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load data
df = pd.read_csv('crofoot.csv')

# Identify variables based on metadata descriptions
# Outcome: 1 if focal won, 0 if other won
outcome = 'm_focal'

# Group sizes (number of individuals in focal and other groups)
size_focal = 'f_other'
size_other = 'win'

# Contest location: distance from each group's home range center
# (focal distance, other distance)
dist_focal = 'm_other'
dist_other = 'n_focal'

# Derived predictors
analysis = df[[outcome, size_focal, size_other, dist_focal, dist_other]].copy()
analysis['size_diff'] = analysis[size_focal] - analysis[size_other]
analysis['size_ratio'] = analysis[size_focal] / analysis[size_other]
analysis['loc_diff'] = analysis[dist_focal] - analysis[dist_other]
analysis['loc_ratio'] = analysis[dist_focal] / analysis[dist_other]

# Standardize continuous predictors for comparability
for col in ['size_diff', 'size_ratio', 'loc_diff', 'loc_ratio']:
    analysis[col + '_z'] = (analysis[col] - analysis[col].mean()) / analysis[col].std(ddof=0)

# Logistic regression: win ~ size_diff + loc_diff
X = analysis[['size_diff_z', 'loc_diff_z']]
X = sm.add_constant(X)
y = analysis[outcome]
model = sm.Logit(y, X).fit(disp=False)

# Alternative model with size_ratio and loc_ratio
X2 = analysis[['size_ratio_z', 'loc_ratio_z']]
X2 = sm.add_constant(X2)
model2 = sm.Logit(y, X2).fit(disp=False)

# Compute odds ratios and 95% CIs for model 1
params = model.params
conf = model.conf_int()
conf.columns = ['2.5%', '97.5%']
odds = np.exp(params)
conf_odds = np.exp(conf)

# Basic descriptive stats
summary_stats = {
    'n': int(len(analysis)),
    'win_rate': float(analysis[outcome].mean()),
    'size_diff_mean': float(analysis['size_diff'].mean()),
    'loc_diff_mean': float(analysis['loc_diff'].mean())
}

# Grouped means by win outcome
means_by_win = analysis.groupby(outcome)[['size_diff', 'loc_diff']].mean().to_dict()

# VIF for predictors
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Print results
print('Logit model: win ~ size_diff_z + loc_diff_z')
print(model.summary())
print('\nOdds ratios (standardized predictors):')
print(pd.DataFrame({'odds_ratio': odds, '2.5%': conf_odds['2.5%'], '97.5%': conf_odds['97.5%']}))
print('\nLogit model: win ~ size_ratio_z + loc_ratio_z')
print(model2.summary())
print('\nSummary stats:', summary_stats)
print('\nMeans by win (0=loss, 1=win):', means_by_win)
print('\nVIF:', vif)
