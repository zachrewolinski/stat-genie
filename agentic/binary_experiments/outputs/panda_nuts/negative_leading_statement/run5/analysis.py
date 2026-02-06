import pandas as pd
import statsmodels.formula.api as smf

# Load data
(df := pd.read_csv('panda_nuts.csv'))

# Define efficiency as nuts opened per second
# (Higher means more nuts cracked per unit time.)
df['efficiency'] = df['nuts_opened'] / df['seconds']

print('Rows:', len(df))
print('Efficiency summary (nuts/second):')
print(df['efficiency'].describe())
print('\nCategory counts:')
print(df[['sex', 'help', 'hammer']].apply(lambda s: s.value_counts()).fillna(0).astype(int))

# Primary model: fixed effects only (age, sex, help)
ols = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit()
print('\nOLS model: efficiency ~ age + sex + help')
print(ols.summary())
print('OLS p-values:')
print(ols.pvalues)

# Random intercept for individual chimpanzee to account for repeated measures
try:
    mixed = smf.mixedlm('efficiency ~ age + C(sex) + C(help)', data=df, groups=df['chimpanzee']).fit(reml=False)
    print('\nMixedLM model (random intercept by chimpanzee):')
    print(mixed.summary())
except Exception as exc:
    mixed = None
    print('\nMixedLM failed:', exc)

# Robustness: include hammer type as a control
ols_hammer = smf.ols('efficiency ~ age + C(sex) + C(help) + C(hammer)', data=df).fit()
print('\nOLS model with hammer control: efficiency ~ age + sex + help + hammer')
print(ols_hammer.summary())
print('OLS + hammer p-values:')
print(ols_hammer.pvalues)
