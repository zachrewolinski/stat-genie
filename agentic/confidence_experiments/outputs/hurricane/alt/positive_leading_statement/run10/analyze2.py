import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

_df = pd.read_csv('hurricane.csv')

# Poisson / NegBin for count data
# Add small constant to avoid issues if needed (not for Poisson)

# Poisson with controls
poisson = smf.glm(
    'alldeaths ~ masfem + wind + min + category + ndam15 + year',
    data=_df,
    family=sm.families.Poisson()
).fit(cov_type='HC3')
print(poisson.summary())

# Poisson with interaction
poisson_int = smf.glm(
    'alldeaths ~ masfem * wind + min + category + ndam15 + year',
    data=_df,
    family=sm.families.Poisson()
).fit(cov_type='HC3')
print(poisson_int.summary())

# Negative binomial with controls
nb = smf.glm(
    'alldeaths ~ masfem + wind + min + category + ndam15 + year',
    data=_df,
    family=sm.families.NegativeBinomial(alpha=1.0)
).fit(cov_type='HC3')
print(nb.summary())

# Negative binomial with interaction
nb_int = smf.glm(
    'alldeaths ~ masfem * wind + min + category + ndam15 + year',
    data=_df,
    family=sm.families.NegativeBinomial(alpha=1.0)
).fit(cov_type='HC3')
print(nb_int.summary())
