import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
raw = pd.read_csv('panda_nuts.csv')

# Infer columns based on known patterns
# nuts_opened column actually holds sex (m/f)
# sex column holds hammer type
# help column holds nuts opened count
# chimpanzee column holds session duration (seconds)
# seconds column holds help indicator (y/N)

df = raw.copy()

# Map variables to meaningful names

df['age_years'] = df['age']

df['sex_mf'] = df['nuts_opened']  # m/f

df['helped'] = df['seconds'].str.upper().map({'Y': 1, 'N': 0})

# Nuts opened count and duration in seconds

df['nuts_opened_count'] = df['help']

df['duration_sec'] = df['chimpanzee']

# Compute efficiency (nuts per second)

df['efficiency'] = df['nuts_opened_count'] / df['duration_sec']

# Basic sanity checks

print('Rows', len(df))
print(df[['age_years','sex_mf','helped','nuts_opened_count','duration_sec','efficiency']].head())

# Drop any rows with missing values

df = df.dropna(subset=['age_years','sex_mf','helped','nuts_opened_count','duration_sec'])

# Poisson GLM with log link and offset for duration

df['log_duration'] = np.log(df['duration_sec'])

poisson_model = smf.glm(
    formula='nuts_opened_count ~ age_years + C(sex_mf) + helped',
    data=df,
    family=sm.families.Poisson(),
    offset=df['log_duration']
).fit()

print('\nPoisson GLM')
print(poisson_model.summary())

# Check dispersion

dispersion = poisson_model.deviance / poisson_model.df_resid
print('Dispersion', dispersion)

# If overdispersed, fit negative binomial
if dispersion > 1.5:
    nb_model = smf.glm(
        formula='nuts_opened_count ~ age_years + C(sex_mf) + helped',
        data=df,
        family=sm.families.NegativeBinomial(alpha=1.0),
        offset=df['log_duration']
    ).fit()
    print('\nNegative Binomial GLM')
    print(nb_model.summary())
else:
    nb_model = None

# Also run linear regression on log efficiency (add small epsilon)

df['log_eff'] = np.log(df['efficiency'] + 1e-6)
ols = smf.ols('log_eff ~ age_years + C(sex_mf) + helped', data=df).fit()
print('\nOLS log-efficiency')
print(ols.summary())

# Save key stats to json-ish output for reading

params = poisson_model.params
conf = poisson_model.conf_int()

results = []
for param in params.index:
    if param == 'Intercept':
        continue
    rr = np.exp(params[param])
    rr_low = np.exp(conf.loc[param, 0])
    rr_high = np.exp(conf.loc[param, 1])
    results.append((param, poisson_model.pvalues[param], rr, rr_low, rr_high))

print('\nRate ratios (Poisson)')
for param, p, rr, low, high in results:
    print(param, 'p=', p, 'RR=', rr, 'CI', low, high)

# If NB model exists, show pvalues
if nb_model is not None:
    print('\nNB pvalues')
    print(nb_model.pvalues)

