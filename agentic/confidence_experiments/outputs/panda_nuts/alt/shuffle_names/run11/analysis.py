import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
csv_path = 'panda_nuts.csv'
df = pd.read_csv(csv_path)

# Map shuffled column names to actual variables based on info.json descriptions
# age -> chimpanzee ID, hammer -> age (years), nuts_opened -> sex, sex -> hammer type
# help -> nuts opened count, chimpanzee -> seconds, seconds -> received help (y/n)

df = df.rename(columns={
    'age': 'chimp_id',
    'hammer': 'age',
    'nuts_opened': 'sex',
    'sex': 'hammer_type',
    'help': 'nuts_opened',
    'chimpanzee': 'seconds',
    'seconds': 'help'
})

# Clean help indicator
help_map = {'y': 1, 'Y': 1, 'n': 0, 'N': 0}
df['help'] = df['help'].map(help_map)

# Drop rows with missing/invalid values
initial_rows = len(df)
df = df.dropna(subset=['age', 'sex', 'help', 'nuts_opened', 'seconds'])
df = df[df['seconds'] > 0]

# Efficiency: nuts per second
# Use Poisson GLM with offset for exposure (seconds)

# Ensure categorical vars

df['sex'] = df['sex'].astype('category')
df['help'] = df['help'].astype('category')

# Model
formula = 'nuts_opened ~ age + C(sex) + C(help)'
model = smf.glm(formula=formula,
                data=df,
                family=sm.families.Poisson(),
                offset=np.log(df['seconds']))

result = model.fit(cov_type='HC3')

# Compute incidence rate ratios and CI
params = result.params
conf = result.conf_int()
irr = np.exp(params)
irr_ci = np.exp(conf)

# Overdispersion check
# Pearson chi2 / df
pearson_chi2 = result.pearson_chi2
pearson_df = result.df_resid
overdispersion = pearson_chi2 / pearson_df if pearson_df > 0 else np.nan

print('Rows used:', len(df), 'of', initial_rows)
print('\nModel summary (Poisson with offset):')
print(result.summary())

print('\nIncidence rate ratios (IRR) and 95% CI:')
for name in params.index:
    print(f"{name:>15}  IRR={irr[name]:.3f}  CI=({irr_ci.loc[name,0]:.3f}, {irr_ci.loc[name,1]:.3f})  p={result.pvalues[name]:.4f}")

print('\nOverdispersion (Pearson chi2 / df):', overdispersion)

# Also fit Negative Binomial as sensitivity if overdispersion
try:
    nb_model = smf.glm(formula=formula,
                       data=df,
                       family=sm.families.NegativeBinomial(),
                       offset=np.log(df['seconds']))
    nb_result = nb_model.fit(cov_type='HC3')
    print('\nNegative Binomial model (sensitivity):')
    print(nb_result.summary())
    nb_params = nb_result.params
    nb_conf = nb_result.conf_int()
    nb_irr = np.exp(nb_params)
    nb_irr_ci = np.exp(nb_conf)
    print('\nNB IRR and 95% CI:')
    for name in nb_params.index:
        print(f"{name:>15}  IRR={nb_irr[name]:.3f}  CI=({nb_irr_ci.loc[name,0]:.3f}, {nb_irr_ci.loc[name,1]:.3f})  p={nb_result.pvalues[name]:.4f}")
except Exception as e:
    print('NB model failed:', e)
