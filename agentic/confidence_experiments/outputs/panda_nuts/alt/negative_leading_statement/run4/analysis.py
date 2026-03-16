import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv("panda_nuts.csv")

# Basic cleaning
# Normalize help to lowercase y/n
# The data has 'y' and 'N' samples; convert to lower

df['help'] = df['help'].astype(str).str.strip().str.lower()
# sex lower

df['sex'] = df['sex'].astype(str).str.strip().str.lower()

# Compute efficiency: nuts opened per second

df['efficiency'] = df['nuts_opened'] / df['seconds']

# Avoid log(0) issues

df['efficiency_log'] = np.log(df['efficiency'] + 1e-6)

# Poisson GLM with offset for time (rate model)
# Model nuts_opened ~ age + sex + help, offset log(seconds)

formula = 'nuts_opened ~ age + C(sex) + C(help)'

poisson_model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df['seconds'])
).fit(cov_type='HC3')

# Negative Binomial GLM (overdispersion check)
nb_model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.NegativeBinomial(alpha=1.0),
    offset=np.log(df['seconds'])
).fit(cov_type='HC3')

# OLS on log efficiency as a robustness check
ols_model = smf.ols('efficiency_log ~ age + C(sex) + C(help)', data=df).fit(cov_type='HC3')

# Summaries

results = {
    'n': len(df),
    'efficiency_mean': df['efficiency'].mean(),
    'efficiency_std': df['efficiency'].std(),
    'poisson_params': poisson_model.params.to_dict(),
    'poisson_pvalues': poisson_model.pvalues.to_dict(),
    'poisson_aic': poisson_model.aic,
    'nb_params': nb_model.params.to_dict(),
    'nb_pvalues': nb_model.pvalues.to_dict(),
    'nb_aic': nb_model.aic,
    'ols_params': ols_model.params.to_dict(),
    'ols_pvalues': ols_model.pvalues.to_dict(),
    'ols_r2': ols_model.rsquared,
}

print(results)
