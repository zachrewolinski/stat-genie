import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Basic cleaning
# Ensure categorical types
for col in ['sex', 'help', 'hammer']:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Create efficiency measures
# rate per second and per minute
# Avoid division by zero (seconds min is 2.5 so safe)
df['rate_per_sec'] = df['nuts_opened'] / df['seconds']
df['rate_per_min'] = df['rate_per_sec'] * 60.0

# Log offset for counts
# Use Poisson and Negative Binomial GLM with offset log(seconds)
# Model nuts_opened ~ age + sex + help (categorical)

# Poisson
poisson_model = smf.glm(
    formula='nuts_opened ~ age + C(sex) + C(help)',
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df['seconds'])
).fit()

# Check overdispersion
pearson_chi2 = ((poisson_model.resid_pearson**2).sum())
df_resid = poisson_model.df_resid
overdispersion = pearson_chi2 / df_resid

# Negative Binomial (use alpha from Poisson as starting point)
# Use NB2; statsmodels uses alpha dispersion parameter
nb_model = smf.glm(
    formula='nuts_opened ~ age + C(sex) + C(help)',
    data=df,
    family=sm.families.NegativeBinomial(alpha=1.0),
    offset=np.log(df['seconds'])
).fit()

# Collect results
results = {
    'poisson': {
        'params': poisson_model.params.to_dict(),
        'pvalues': poisson_model.pvalues.to_dict(),
        'aic': poisson_model.aic,
        'overdispersion': float(overdispersion),
    },
    'neg_bin': {
        'params': nb_model.params.to_dict(),
        'pvalues': nb_model.pvalues.to_dict(),
        'aic': nb_model.aic,
    }
}

# Also run linear model on rate per minute for interpretability
lm = smf.ols('rate_per_min ~ age + C(sex) + C(help)', data=df).fit()
results['ols_rate'] = {
    'params': lm.params.to_dict(),
    'pvalues': lm.pvalues.to_dict(),
    'r2': lm.rsquared,
    'adj_r2': lm.rsquared_adj,
}

# Print results for inspection
print('N rows:', len(df))
print('Overdispersion (Poisson):', overdispersion)
print('\nPoisson p-values:')
print(poisson_model.pvalues)
print('\nNegBin p-values:')
print(nb_model.pvalues)
print('\nOLS rate p-values:')
print(lm.pvalues)

# Save summary results to CSV-like text for easy reading if needed
import json
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)
