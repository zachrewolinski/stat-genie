import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = 'crofoot.csv'
df = pd.read_csv(csv_path)

print('Columns:', df.columns.tolist())
print('Shape:', df.shape)
print(df.head())

# Focus on relevant columns
cols = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other']
missing_cols = [c for c in cols if c not in df.columns]
print('Missing cols:', missing_cols)

sub = df[cols].copy()
print(sub.describe())

# Compute relative group size and contest location metrics
sub['rel_size'] = sub['n_focal'] - sub['n_other']
# Ratio (avoid divide by zero)
sub['size_ratio'] = sub['n_focal'] / sub['n_other']

# Contest location: positive means contest closer to focal group center (other farther)
sub['loc_adv'] = sub['dist_other'] - sub['dist_focal']

print(sub[['rel_size','size_ratio','loc_adv']].describe())

# Check distributions and any missing values
print('Missing values:', sub.isna().sum())

# Logistic regression using GLM (binomial)
# Standardize predictors for interpretability
for col in ['rel_size','loc_adv']:
    sub[f'z_{col}'] = (sub[col] - sub[col].mean()) / sub[col].std(ddof=0)

model = smf.glm('win ~ z_rel_size + z_loc_adv', data=sub, family=sm.families.Binomial()).fit()
print(model.summary())

# Alternative model using size_ratio
sub['z_size_ratio'] = (sub['size_ratio'] - sub['size_ratio'].mean()) / sub['size_ratio'].std(ddof=0)
model_ratio = smf.glm('win ~ z_size_ratio + z_loc_adv', data=sub, family=sm.families.Binomial()).fit()
print(model_ratio.summary())

# Single predictor models
model_size = smf.glm('win ~ z_rel_size', data=sub, family=sm.families.Binomial()).fit()
model_loc = smf.glm('win ~ z_loc_adv', data=sub, family=sm.families.Binomial()).fit()
print(model_size.summary())
print(model_loc.summary())

# Compute odds ratios for main model
params = model.params
conf = model.conf_int()
odds = np.exp(params)
conf_odds = np.exp(conf)
print('Odds ratios (main model):')
print(pd.DataFrame({'odds_ratio': odds, 'ci_low': conf_odds[0], 'ci_high': conf_odds[1]}))

# Predictive probabilities for illustrative changes
# Use mean values and +/-1 SD changes
mean_row = { 'z_rel_size': 0.0, 'z_loc_adv': 0.0 }
mean_prob = model.predict(pd.DataFrame([mean_row]))[0]
plus_size = model.predict(pd.DataFrame([{ 'z_rel_size': 1.0, 'z_loc_adv': 0.0 }]))[0]
minus_size = model.predict(pd.DataFrame([{ 'z_rel_size': -1.0, 'z_loc_adv': 0.0 }]))[0]
plus_loc = model.predict(pd.DataFrame([{ 'z_rel_size': 0.0, 'z_loc_adv': 1.0 }]))[0]
minus_loc = model.predict(pd.DataFrame([{ 'z_rel_size': 0.0, 'z_loc_adv': -1.0 }]))[0]

print('Predicted prob at mean:', mean_prob)
print('Predicted prob +/-1 SD rel_size:', minus_size, plus_size)
print('Predicted prob +/-1 SD loc_adv:', minus_loc, plus_loc)
