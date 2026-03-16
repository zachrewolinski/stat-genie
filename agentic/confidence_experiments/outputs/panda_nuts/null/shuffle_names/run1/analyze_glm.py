import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

analysis = pd.DataFrame({
    'age_years': df['hammer'].astype(float),
    'sex': df['nuts_opened'].astype(str),
    'help_received': df['seconds'].astype(str),
    'nuts_opened': df['help'].astype(float),
    'duration_sec': df['chimpanzee'].astype(float),
})

# Poisson regression with offset for exposure time
analysis['log_duration'] = np.log(analysis['duration_sec'])

poisson = smf.glm(
    formula='nuts_opened ~ age_years + C(sex) + C(help_received)',
    data=analysis,
    family=sm.families.Poisson(),
    offset=analysis['log_duration']
).fit(cov_type='HC3')

print('POISSON')
print(poisson.summary())

# Negative binomial (use NB2 with alpha estimated by statsmodels)
nb = smf.glm(
    formula='nuts_opened ~ age_years + C(sex) + C(help_received)',
    data=analysis,
    family=sm.families.NegativeBinomial(alpha=1.0),
    offset=analysis['log_duration']
).fit(cov_type='HC3')

print('\nNEGATIVE BINOMIAL (alpha fixed=1.0)')
print(nb.summary())

# Dispersion estimate for poisson (Pearson chi2 / df)
pearson_chi2 = sum(poisson.resid_pearson**2)
df_resid = poisson.df_resid
print('\nPoisson dispersion:', pearson_chi2 / df_resid)

