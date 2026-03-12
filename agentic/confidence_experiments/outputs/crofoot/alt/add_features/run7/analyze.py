import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

path = 'crofoot.csv'
df = pd.read_csv(path)

# Derived predictors
# Relative group size: difference and ratio
# Avoid division by zero (not expected)
df['size_diff'] = df['n_focal'] - df['n_other']
df['size_ratio'] = df['n_focal'] / df['n_other']

# Contest location advantage: positive if contest closer to focal home range
# dist_focal = distance from focal home range center; smaller => closer
# Use difference and proportion

df['loc_diff'] = df['dist_other'] - df['dist_focal']
df['loc_prop'] = df['dist_focal'] / (df['dist_focal'] + df['dist_other'])

# Standardize continuous predictors for comparable coefficients
for col in ['size_diff','loc_diff','size_ratio','loc_prop']:
    df[col + '_z'] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

# Logistic regression using size_diff and loc_diff
model1 = smf.logit('win ~ size_diff_z + loc_diff_z', data=df).fit(disp=False)

# Alternative model using ratio and proportion
model2 = smf.logit('win ~ size_ratio_z + loc_prop_z', data=df).fit(disp=False)

# Additive models to see each effect separately
model_size = smf.logit('win ~ size_diff_z', data=df).fit(disp=False)
model_loc = smf.logit('win ~ loc_diff_z', data=df).fit(disp=False)

# Simple comparisons: win rates when focal is closer vs farther/equal
focal_closer = df['dist_focal'] < df['dist_other']
win_rate_closer = df.loc[focal_closer, 'win'].mean()
win_rate_farther = df.loc[~focal_closer, 'win'].mean()

# Correlations (point-biserial) as supplemental
corr_size = np.corrcoef(df['win'], df['size_diff'])[0,1]
corr_loc = np.corrcoef(df['win'], df['loc_diff'])[0,1]

results = {
    'n': int(len(df)),
    'model1': model1.summary2().tables[1].to_dict(),
    'model2': model2.summary2().tables[1].to_dict(),
    'model_size': model_size.summary2().tables[1].to_dict(),
    'model_loc': model_loc.summary2().tables[1].to_dict(),
    'win_rate_closer': win_rate_closer,
    'win_rate_farther': win_rate_farther,
    'corr_size': corr_size,
    'corr_loc': corr_loc,
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
