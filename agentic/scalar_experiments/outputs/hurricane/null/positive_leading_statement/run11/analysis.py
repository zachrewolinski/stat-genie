import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = 'hurricane.csv'
df = pd.read_csv(csv_path)

# Basic cleaning
# log1p for deaths and damages due to skew and zeros

df['log_deaths'] = np.log1p(df['alldeaths'])
df['log_ndam15'] = np.log1p(df['ndam15'])

# Select variables
vars_basic = ['masfem', 'log_deaths']

# Summary stats
summary = df[vars_basic].describe()
print('Summary stats for masfem and log_deaths')
print(summary)

# Correlation
corr = df[['masfem','log_deaths']].corr().iloc[0,1]
print('\nCorrelation (masfem, log_deaths):', corr)

# Unadjusted OLS
model_unadj = smf.ols('log_deaths ~ masfem', data=df).fit(cov_type='HC3')
print('\nUnadjusted OLS (log_deaths ~ masfem)')
print(model_unadj.summary())

# Adjusted model with key storm intensity controls
# Use min pressure (lower pressure = stronger) and wind, category, and log damage as proxies
# Drop rows with missing values for these covariates
model_vars = ['log_deaths','masfem','wind','min','category','log_ndam15']
df_adj = df.dropna(subset=model_vars)

model_adj = smf.ols('log_deaths ~ masfem + wind + min + category + log_ndam15', data=df_adj).fit(cov_type='HC3')
print('\nAdjusted OLS (log_deaths ~ masfem + wind + min + category + log_ndam15)')
print(model_adj.summary())

# Alternative: using binary gender_mf instead of masfem
if 'gender_mf' in df.columns:
    df_adj2 = df.dropna(subset=['log_deaths','gender_mf','wind','min','category','log_ndam15'])
    model_gender = smf.ols('log_deaths ~ gender_mf + wind + min + category + log_ndam15', data=df_adj2).fit(cov_type='HC3')
    print('\nAdjusted OLS with gender_mf')
    print(model_gender.summary())

# Simple robustness: negative binomial on counts with same covariates
try:
    import statsmodels.formula.api as smf2
    nb_model = smf2.glm('alldeaths ~ masfem + wind + min + category + log_ndam15', data=df_adj,
                        family=sm.families.NegativeBinomial()).fit(cov_type='HC3')
    print('\nNegative Binomial (alldeaths)')
    print(nb_model.summary())
except Exception as e:
    print('NB model failed:', e)

# Store key stats for later
print('\nKey results:')
print('Unadj masfem coef:', model_unadj.params['masfem'], 'p=', model_unadj.pvalues['masfem'])
print('Adj masfem coef:', model_adj.params['masfem'], 'p=', model_adj.pvalues['masfem'])
