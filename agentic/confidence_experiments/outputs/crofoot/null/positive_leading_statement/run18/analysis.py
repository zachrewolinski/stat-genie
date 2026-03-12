import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data

df = pd.read_csv('crofoot.csv')

# Create predictors
# Relative group size: focal minus other (positive means focal larger)
df['rel_size'] = df['n_focal'] - df['n_other']

# Location advantage: other distance minus focal distance (positive means contest closer to focal home range center)
df['loc_adv'] = df['dist_other'] - df['dist_focal']

# Standardize predictors for comparability
for col in ['rel_size', 'loc_adv']:
    df[f'z_{col}'] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

# Logistic regression models
results = {}

# Model 1: rel_size only
X1 = sm.add_constant(df[['z_rel_size']])
model1 = sm.Logit(df['win'], X1).fit(disp=0)
results['model1'] = model1

# Model 2: loc_adv only
X2 = sm.add_constant(df[['z_loc_adv']])
model2 = sm.Logit(df['win'], X2).fit(disp=0)
results['model2'] = model2

# Model 3: both predictors
X3 = sm.add_constant(df[['z_rel_size', 'z_loc_adv']])
model3 = sm.Logit(df['win'], X3).fit(disp=0)
results['model3'] = model3

# Print summaries
for name, res in results.items():
    print(f"\n{name}")
    print(res.summary())

# Odds ratios and 95% CI for model3
params = model3.params
conf = model3.conf_int()
conf.columns = ['2.5%', '97.5%']

odds_ratios = np.exp(params)
conf_or = np.exp(conf)

print("\nModel3 odds ratios (exp(beta)) and 95% CI:")
print(pd.DataFrame({'OR': odds_ratios, '2.5%': conf_or['2.5%'], '97.5%': conf_or['97.5%']}))

# Simple descriptive stats
print("\nWin rate overall:", df['win'].mean())
print("Win rate when focal larger (rel_size>0):", df.loc[df['rel_size']>0, 'win'].mean())
print("Win rate when focal smaller (rel_size<0):", df.loc[df['rel_size']<0, 'win'].mean())
print("Win rate when focal closer to center (loc_adv>0):", df.loc[df['loc_adv']>0, 'win'].mean())
print("Win rate when focal farther from center (loc_adv<0):", df.loc[df['loc_adv']<0, 'win'].mean())

# Correlation between predictors
print("\nCorrelation rel_size vs loc_adv:", df['rel_size'].corr(df['loc_adv']))
