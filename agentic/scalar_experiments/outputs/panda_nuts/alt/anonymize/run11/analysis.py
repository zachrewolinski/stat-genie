import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Rename columns for clarity
rename = {
    'feature1': 'id',
    'feature2': 'age',
    'feature3': 'sex',
    'feature4': 'hammer',
    'feature5': 'nuts_opened',
    'feature6': 'duration',
    'feature7': 'help'
}
df = df.rename(columns=rename)

# Efficiency: nuts opened per second
# Avoid division by zero just in case

df['efficiency'] = df['nuts_opened'] / df['duration']

# Basic clean
# Drop rows with missing key variables
key_cols = ['age', 'sex', 'help', 'efficiency']
clean = df.dropna(subset=key_cols).copy()

# OLS with categorical predictors
model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=clean).fit(cov_type='HC3')

# Also check a model with hammer as control (optional)
model_ctrl = smf.ols('efficiency ~ age + C(sex) + C(help) + C(hammer)', data=clean).fit(cov_type='HC3')

# Summaries
print('N rows:', len(clean))
print('Efficiency summary:')
print(clean['efficiency'].describe())

print('\nModel (age + sex + help) robust HC3:')
print(model.summary())

print('\nModel with hammer control robust HC3:')
print(model_ctrl.summary())

# Extract p-values and coefficients
pvals = model.pvalues
coefs = model.params

print('\nCoefficients:')
print(coefs)
print('\nP-values:')
print(pvals)

# For reporting: effect sizes in practical terms
# Predicted difference between sexes and help categories at mean age
mean_age = clean['age'].mean()

# Build two-row dataframe for predictions
levels_sex = sorted(clean['sex'].unique())
levels_help = sorted(clean['help'].unique())

rows = []
for sex in levels_sex:
    for helpv in levels_help:
        rows.append({'age': mean_age, 'sex': sex, 'help': helpv})

pred_df = pd.DataFrame(rows)
pred_df['pred_eff'] = model.predict(pred_df)
print('\nPredicted efficiency at mean age by sex/help (model 1):')
print(pred_df)

# Save key stats for later if needed
model_info = {
    'coef': coefs.to_dict(),
    'pvals': pvals.to_dict(),
    'r2': model.rsquared,
    'adj_r2': model.rsquared_adj,
    'n': len(clean)
}
print('\nModel info dict:')
print(model_info)
