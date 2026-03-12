import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data

df = pd.read_csv('panda_nuts.csv')

# Basic cleaning
# normalize help values to lower-case
if 'help' in df.columns:
    df['help'] = df['help'].astype(str).str.strip().str.lower()

# create efficiency measure: nuts opened per second
# avoid division by zero if any

df['efficiency'] = df['nuts_opened'] / df['seconds']

# drop rows with missing key fields
model_df = df[['efficiency', 'age', 'sex', 'help', 'nuts_opened', 'seconds']].dropna()

# build linear model with categorical sex and help
# use robust standard errors (HC3)

model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=model_df).fit(cov_type='HC3')

# also check alternative with log efficiency to handle skew
model_log = smf.ols('np.log1p(efficiency) ~ age + C(sex) + C(help)', data=model_df).fit(cov_type='HC3')

# anova for overall effects (type II) using statsmodels
from statsmodels.stats.anova import anova_lm

anova = anova_lm(model, typ=2)

print('N rows:', len(model_df))
print(model.summary())
print('\nType II ANOVA:')
print(anova)
print('\nLog model summary:')
print(model_log.summary())
