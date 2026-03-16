import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
csv_path = 'panda_nuts.csv'
df = pd.read_csv(csv_path)

# Clean categorical columns
for col in ['help', 'sex']:
    df[col] = df[col].astype(str).str.strip().str.lower()

# Compute efficiency (nuts per second)
df['efficiency'] = df['nuts_opened'] / df['seconds']

print('Rows:', len(df))
print('Missing values:', df.isna().sum())
print('\nEfficiency summary:')
print(df['efficiency'].describe())

# OLS on efficiency (simple linear model)
ols_formula = 'efficiency ~ age + C(sex) + C(help)'
ols_model = smf.ols(ols_formula, data=df).fit()
print('\nOLS summary (efficiency):')
print(ols_model.summary())

# Robust SE for OLS
ols_robust = ols_model.get_robustcov_results(cov_type='HC3')
print('\nOLS robust (HC3) summary:')
print(ols_robust.summary())

# ANOVA for OLS
anova = sm.stats.anova_lm(ols_model, typ=2)
print('\nANOVA (type II) for OLS:')
print(anova)

# Poisson regression on counts with offset log(seconds)
if (df['seconds'] <= 0).any():
    raise ValueError('Non-positive seconds found')

poisson_formula = 'nuts_opened ~ age + C(sex) + C(help)'
poisson_model = smf.glm(
    poisson_formula,
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df['seconds'])
).fit()
print('\nPoisson GLM summary (count with offset):')
print(poisson_model.summary())

# Robust SE for Poisson to handle overdispersion
try:
    poisson_robust = smf.glm(
        poisson_formula,
        data=df,
        family=sm.families.Poisson(),
        offset=np.log(df['seconds'])
    ).fit(cov_type='HC0')
    print('\nPoisson GLM robust (HC0) summary:')
    print(poisson_robust.summary())
except Exception as e:
    print('Poisson robust fit failed:', e)
    poisson_robust = None

# Likelihood ratio test vs null
null_model = smf.glm(
    'nuts_opened ~ 1',
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df['seconds'])
).fit()

lr_stat = 2 * (poisson_model.llf - null_model.llf)
df_diff = int(poisson_model.df_model - null_model.df_model)
p_lr = stats.chi2.sf(lr_stat, df_diff)
print('\nPoisson LR test vs null:')
print({'lr_stat': lr_stat, 'df': df_diff, 'p_value': p_lr})

# Check overdispersion (Pearson chi2 / df)
pearson_chi2 = poisson_model.pearson_chi2
od = pearson_chi2 / poisson_model.df_resid
print('\nOverdispersion ratio:', od)

# Negative Binomial (discrete) with offset, to handle overdispersion
try:
    nb_model = smf.negativebinomial(
        poisson_formula,
        data=df,
        offset=np.log(df['seconds'])
    ).fit(disp=False)
    print('\nNegative Binomial (discrete) summary:')
    print(nb_model.summary())
except Exception as e:
    print('Negative Binomial model failed:', e)
    nb_model = None

print('\nCoefficients (Poisson):')
print(poisson_model.params)
print('\nP-values (Poisson):')
print(poisson_model.pvalues)

if poisson_robust is not None:
    print('\nP-values (Poisson robust):')
    print(poisson_robust.pvalues)

print('\nCoefficients (OLS):')
print(ols_model.params)
print('\nP-values (OLS):')
print(ols_model.pvalues)

if nb_model is not None:
    print('\nP-values (Negative Binomial):')
    print(nb_model.pvalues)
