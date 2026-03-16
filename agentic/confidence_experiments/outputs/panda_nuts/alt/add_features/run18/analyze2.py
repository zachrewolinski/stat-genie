import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('panda_nuts.csv')

# Prep
cols = ['age','sex','help','nuts_opened','seconds']
df = df[cols].copy()

for c in ['age','nuts_opened','seconds']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Normalize help

def normalize_help(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in ['y','yes','1','true','t']:
        return 'y'
    if s in ['n','no','0','false','f']:
        return 'n'
    return s


df['help_norm'] = df['help'].apply(normalize_help)

# Filter

df = df[df['seconds'] > 0].copy()

# Drop missing

df_model = df.dropna(subset=['age','sex','help_norm','nuts_opened','seconds']).copy()

# categorical

df_model['sex'] = df_model['sex'].astype('category')

df_model['help_norm'] = df_model['help_norm'].astype('category')

# Poisson GLM for counts with log(seconds) offset

formula = 'nuts_opened ~ age + C(sex) + C(help_norm)'

poisson_model = smf.glm(formula=formula, data=df_model,
                        family=sm.families.Poisson(),
                        offset=np.log(df_model['seconds'])).fit(cov_type='HC3')

# Check overdispersion

mu = poisson_model.mu
var = ((df_model['nuts_opened'] - mu) ** 2 - mu).sum() / (len(df_model) - poisson_model.df_model - 1)

# Negative binomial (NB2) with log link

nb_model = smf.glm(formula=formula, data=df_model,
                   family=sm.families.NegativeBinomial(alpha=1.0),
                   offset=np.log(df_model['seconds'])).fit(cov_type='HC3')

print('Poisson params and p-values:')
print(pd.DataFrame({'coef': poisson_model.params, 'p': poisson_model.pvalues, 'std_err': poisson_model.bse}))
print('Poisson overdispersion estimate:', var)
print('\nNegative binomial params and p-values:')
print(pd.DataFrame({'coef': nb_model.params, 'p': nb_model.pvalues, 'std_err': nb_model.bse}))

# Rate ratios
print('\nPoisson rate ratios:')
print(np.exp(poisson_model.params))
print('\nNB rate ratios:')
print(np.exp(nb_model.params))
