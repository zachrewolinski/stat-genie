import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('crofoot.csv')

# Focus on relevant columns; drop rows with missing values in these fields
cols = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other']
df = _df[cols].dropna().copy()

# Derived predictors
# Relative group size: focal minus other (positive means focal larger)
df['rel_size'] = df['n_focal'] - df['n_other']
# Location advantage: positive means contest is closer to focal group's center
# (i.e., other is farther from its own center than focal is from its own)
df['loc_adv'] = df['dist_other'] - df['dist_focal']

# Logistic regression: win ~ rel_size + loc_adv
X = df[['rel_size', 'loc_adv']]
X = sm.add_constant(X)
y = df['win']

model = sm.Logit(y, X)
result = model.fit(disp=False)

# Print key outputs
print('N:', len(df))
print('Win rate:', df['win'].mean())
print('\nPredictor summary:')
print(df[['rel_size', 'loc_adv']].describe())
print('\nLogit results (win ~ rel_size + loc_adv):')
print(result.summary())

# Compute odds ratios for interpretability
params = result.params
conf = result.conf_int()
conf.columns = ['2.5%', '97.5%']

odds = np.exp(params)
conf_odds = np.exp(conf)

or_table = pd.DataFrame({
    'odds_ratio': odds,
    'or_2.5%': conf_odds['2.5%'],
    'or_97.5%': conf_odds['97.5%'],
    'p_value': result.pvalues,
})

print('\nOdds ratios (exp(coef)) with 95% CI:')
print(or_table)
