import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'crofoot.csv'
df = pd.read_csv(path)

print('shape', df.shape)
print('columns', list(df.columns))
print(df.head())

# Identify variables relevant to question
# relative group size: use n_focal vs n_other (group sizes)
# contest location: compare distances to home range centers; create location advantage = dist_other - dist_focal
# win is outcome (1 focal wins)

# Clean / derive
if 'n_focal' in df.columns and 'n_other' in df.columns:
    df['rel_group_size'] = df['n_focal'] - df['n_other']
else:
    df['rel_group_size'] = np.nan

if 'dist_focal' in df.columns and 'dist_other' in df.columns:
    # positive means contest closer to focal home range center (focal is closer)
    df['loc_advantage'] = df['dist_other'] - df['dist_focal']
else:
    df['loc_advantage'] = np.nan

# Drop rows with missing in required fields
required = ['win', 'rel_group_size', 'loc_advantage']
clean = df.dropna(subset=required)

print('clean shape', clean.shape)

# Descriptive stats
print(clean[required].describe())

# Logistic regression: win ~ rel_group_size + loc_advantage
# Standardize predictors for comparability
clean = clean.copy()
clean['rel_group_size_z'] = (clean['rel_group_size'] - clean['rel_group_size'].mean()) / clean['rel_group_size'].std(ddof=0)
clean['loc_advantage_z'] = (clean['loc_advantage'] - clean['loc_advantage'].mean()) / clean['loc_advantage'].std(ddof=0)

model = smf.logit('win ~ rel_group_size_z + loc_advantage_z', data=clean).fit(disp=False)
print(model.summary())

# Also evaluate each predictor individually
model_size = smf.logit('win ~ rel_group_size_z', data=clean).fit(disp=False)
model_loc = smf.logit('win ~ loc_advantage_z', data=clean).fit(disp=False)

print('size only', model_size.summary())
print('loc only', model_loc.summary())

# Compute odds ratios with CIs
params = model.params
conf = model.conf_int()
conf.columns = ['2.5%', '97.5%']
OR = np.exp(params)
OR_ci = np.exp(conf)
print('OR', OR)
print('OR CI', OR_ci)

# Predictive accuracy (simple)
pred = (model.predict(clean) >= 0.5).astype(int)
acc = (pred == clean['win']).mean()
print('accuracy', acc)

# Save key stats to json for later
import json
out = {
    'n': int(clean.shape[0]),
    'coef': model.params.to_dict(),
    'pvalues': model.pvalues.to_dict(),
    'odds_ratio': OR.to_dict(),
    'odds_ratio_ci': OR_ci.to_dict(),
    'accuracy': float(acc)
}
with open('analysis_results.json', 'w') as f:
    json.dump(out, f, indent=2)
