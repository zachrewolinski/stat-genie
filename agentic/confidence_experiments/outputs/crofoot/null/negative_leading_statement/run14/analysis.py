import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('crofoot.csv')

# Create relative predictors
_df['size_diff'] = _df['n_focal'] - _df['n_other']
# Positive dist_diff means the contest is closer to the focal group's home-range center
_df['dist_diff'] = _df['dist_other'] - _df['dist_focal']

# Standardize predictors for numerical stability / effect comparability
for col in ['size_diff', 'dist_diff']:
    _df[col + '_z'] = (_df[col] - _df[col].mean()) / _df[col].std(ddof=0)

# Model 1: win ~ size_diff + dist_diff (standardized)
X1 = sm.add_constant(_df[['size_diff_z', 'dist_diff_z']])
model1 = sm.GLM(_df['win'], X1, family=sm.families.Binomial())
res1 = model1.fit()

# Model 2: win ~ size_diff + dist_diff (raw)
X2 = sm.add_constant(_df[['size_diff', 'dist_diff']])
model2 = sm.GLM(_df['win'], X2, family=sm.families.Binomial())
res2 = model2.fit()

# Model 3: alternative parameterization with size ratio
_df['size_ratio'] = _df['n_focal'] / _df['n_other']
_df['size_ratio_z'] = (_df['size_ratio'] - _df['size_ratio'].mean()) / _df['size_ratio'].std(ddof=0)
X3 = sm.add_constant(_df[['size_ratio_z', 'dist_diff_z']])
model3 = sm.GLM(_df['win'], X3, family=sm.families.Binomial())
res3 = model3.fit()

# Summaries
summary = {
    'n_rows': int(_df.shape[0]),
    'win_rate': float(_df['win'].mean()),
    'model1_params': res1.params.to_dict(),
    'model1_pvalues': res1.pvalues.to_dict(),
    'model1_or': np.exp(res1.params).to_dict(),
    'model2_params': res2.params.to_dict(),
    'model2_pvalues': res2.pvalues.to_dict(),
    'model2_or': np.exp(res2.params).to_dict(),
    'model3_params': res3.params.to_dict(),
    'model3_pvalues': res3.pvalues.to_dict(),
    'model3_or': np.exp(res3.params).to_dict(),
}

print(summary)
