import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv('panda_nuts.csv')
print(df.head())
print(df.describe(include='all'))

# create efficiency: nuts per second
# handle seconds zero? check min
print('seconds min', df['seconds'].min())

df['efficiency'] = df['nuts_opened'] / df['seconds']

print(df['efficiency'].describe())

# treat categorical
print(df['sex'].value_counts())
print(df['help'].value_counts())

# Linear regression for efficiency
model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit()
print(model.summary())

# Alternative: model nuts_opened with offset seconds? Poisson or negative binomial
# Use GLM Poisson with log(seconds) offset
model_pois = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=df,
                     family=sm.families.Poisson(),
                     offset=np.log(df['seconds'])).fit()
print(model_pois.summary())

# Overdispersion check: pearson chi2 / df
pearson = model_pois.pearson_chi2 / model_pois.df_resid
print('Poisson dispersion', pearson)

# Negative binomial as robustness
model_nb = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=df,
                   family=sm.families.NegativeBinomial(alpha=1.0),
                   offset=np.log(df['seconds'])).fit()
print(model_nb.summary())

