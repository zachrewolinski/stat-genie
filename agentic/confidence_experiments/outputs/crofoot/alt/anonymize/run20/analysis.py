import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'crofoot.csv'
df = pd.read_csv(path)

# Define variables
# feature4: win (1 focal won, 0 other won)
# feature5: focal distance from center
# feature6: other distance from center
# feature7: focal group size
# feature8: other group size

df['win'] = df['feature4']
# Relative group size (difference and ratio)
df['size_diff'] = df['feature7'] - df['feature8']
df['size_ratio'] = df['feature7'] / df['feature8']
# Relative location (positive if focal is closer to its center than other is to its center)
df['loc_diff'] = df['feature6'] - df['feature5']  # other distance minus focal distance
# Smaller distance = closer to center. So positive loc_diff => other is farther from its center.
# For ratio, >1 means focal farther than other.
df['loc_ratio'] = df['feature5'] / df['feature6']

# Logistic regression models
# Add constant for non-formula models
X1 = sm.add_constant(df[['size_diff']])
X2 = sm.add_constant(df[['loc_diff']])
X3 = sm.add_constant(df[['size_diff', 'loc_diff']])

models = {}
models['size_only'] = sm.Logit(df['win'], X1).fit(disp=False)
models['loc_only'] = sm.Logit(df['win'], X2).fit(disp=False)
models['size_loc'] = sm.Logit(df['win'], X3).fit(disp=False)

# Also check ratio versions
X4 = sm.add_constant(df[['size_ratio']])
X5 = sm.add_constant(df[['loc_ratio']])
X6 = sm.add_constant(df[['size_ratio', 'loc_ratio']])
models['size_ratio_only'] = sm.Logit(df['win'], X4).fit(disp=False)
models['loc_ratio_only'] = sm.Logit(df['win'], X5).fit(disp=False)
models['size_loc_ratio'] = sm.Logit(df['win'], X6).fit(disp=False)

# Summaries
out = {}
for name, m in models.items():
    params = m.params
    pvalues = m.pvalues
    conf = m.conf_int()
    out[name] = {
        'n': int(m.nobs),
        'params': params.to_dict(),
        'pvalues': pvalues.to_dict(),
        'conf_int': conf.to_dict(),
        'llf': float(m.llf),
        'aic': float(m.aic),
        'pseudo_r2': float(m.prsquared),
    }

# Descriptive stats
summary = {
    'n': int(len(df)),
    'win_rate': float(df['win'].mean()),
    'size_diff_mean': float(df['size_diff'].mean()),
    'loc_diff_mean': float(df['loc_diff'].mean()),
}

result = {'summary': summary, 'models': out}
print(json.dumps(result, indent=2))
