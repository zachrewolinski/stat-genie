import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'crofoot.csv'
df = pd.read_csv(path)

# Rename columns for clarity
cols = {
    'feature4': 'focal_win',
    'feature5': 'focal_dist',
    'feature6': 'other_dist',
    'feature7': 'focal_size',
    'feature8': 'other_size'
}

df = df.rename(columns=cols)

# Derived variables
# Relative group size: focal - other (positive means focal larger)
df['rel_size'] = df['focal_size'] - df['other_size']
# Contest location advantage: other distance - focal distance (positive means contest closer to focal center)
df['loc_adv'] = df['other_dist'] - df['focal_dist']

# Basic summaries
print('Rows:', len(df))
print(df[['focal_win','focal_size','other_size','rel_size','focal_dist','other_dist','loc_adv']].describe())

# Logistic regression
# Model with rel_size and loc_adv
model = smf.logit('focal_win ~ rel_size + loc_adv', data=df).fit(disp=False)
print('\nLogit model: focal_win ~ rel_size + loc_adv')
print(model.summary())

# Also standardize predictors for effect sizes
for col in ['rel_size','loc_adv']:
    df[col+'_z'] = (df[col] - df[col].mean())/df[col].std(ddof=0)

model_z = smf.logit('focal_win ~ rel_size_z + loc_adv_z', data=df).fit(disp=False)
print('\nLogit model (z):')
print(model_z.summary())

# Add alternative definition for location: focal distance minus other distance
# (negative means closer to focal center)
# but loc_adv already other - focal. We'll test if sign holds

# Univariate models
model_size = smf.logit('focal_win ~ rel_size', data=df).fit(disp=False)
model_loc = smf.logit('focal_win ~ loc_adv', data=df).fit(disp=False)
print('\nUnivariate: rel_size')
print(model_size.summary())
print('\nUnivariate: loc_adv')
print(model_loc.summary())

# Predicted probabilities for interpretation
# Compute odds ratios
params = model.params
conf = model.conf_int()
conf.columns = ['2.5%','97.5%']

or_table = pd.DataFrame({
    'coef': params,
    'OR': np.exp(params),
    'OR_2.5%': np.exp(conf['2.5%']),
    'OR_97.5%': np.exp(conf['97.5%']),
    'p': model.pvalues
})
print('\nOdds ratios (main model):')
print(or_table)

# Save for later if needed
or_table.to_csv('or_table.csv', index=True)
