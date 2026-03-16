import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Basic cleaning
# Ensure numeric columns
numeric_cols = ['masfem','masfem_mturk','gender_mf','min','category','alldeaths','wind','ndam','ndam15','elapsedyrs','year']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing key variables
key_cols = ['alldeaths','masfem','wind','min','category']
base = df.dropna(subset=key_cols).copy()

base['log_deaths'] = np.log1p(base['alldeaths'])

# OLS models
models = {}

formula1 = 'log_deaths ~ masfem'
models['ols_1'] = smf.ols(formula1, data=base).fit(cov_type='HC3')

formula2 = 'log_deaths ~ masfem + wind + min + category'
models['ols_2'] = smf.ols(formula2, data=base).fit(cov_type='HC3')

formula3 = 'log_deaths ~ masfem + wind + min + category + elapsedyrs'
models['ols_3'] = smf.ols(formula3, data=base).fit(cov_type='HC3')

# Gender binary
formula4 = 'log_deaths ~ gender_mf + wind + min + category + elapsedyrs'
models['ols_4'] = smf.ols(formula4, data=base).fit(cov_type='HC3')

# GLM Poisson on deaths (counts), with robust SEs
# add small epsilon to avoid issues? Poisson handles zeros.
formula5 = 'alldeaths ~ masfem + wind + min + category + elapsedyrs'
models['glm_pois'] = smf.glm(formula5, data=base, family=sm.families.Poisson()).fit(cov_type='HC3')

# GLM Negative Binomial (to handle overdispersion)
models['glm_nb'] = smf.glm(formula5, data=base, family=sm.families.NegativeBinomial(alpha=1.0)).fit(cov_type='HC3')

# Correlations
corr_masfem_deaths = base[['masfem','alldeaths']].corr().iloc[0,1]
corr_masfem_log = base[['masfem','log_deaths']].corr().iloc[0,1]

# Summary table for key coefficients
summary_rows = []
for name, model in models.items():
    if 'masfem' in model.params.index:
        coef = model.params['masfem']
        se = model.bse['masfem']
        pval = model.pvalues['masfem']
        summary_rows.append([name, 'masfem', coef, se, pval])
    if 'gender_mf' in model.params.index:
        coef = model.params['gender_mf']
        se = model.bse['gender_mf']
        pval = model.pvalues['gender_mf']
        summary_rows.append([name, 'gender_mf', coef, se, pval])

summary_df = pd.DataFrame(summary_rows, columns=['model','term','coef','se','pval'])

# Print results
print('Rows:', len(base))
print('Correlation masfem vs alldeaths:', corr_masfem_deaths)
print('Correlation masfem vs log1p(alldeaths):', corr_masfem_log)
print('\nKey coefficient summaries:')
print(summary_df.to_string(index=False))

# Add a simple effect size interpretation for OLS log model with full controls
m = models['ols_3']
if 'masfem' in m.params.index:
    coef = m.params['masfem']
    # percent change in deaths per 1 unit increase in masfem in log scale
    pct = (np.exp(coef) - 1) * 100
    print(f"\nOLS (log deaths) masfem coef {coef:.4f} implies approx {pct:.2f}% change in deaths per unit (if significant).")

