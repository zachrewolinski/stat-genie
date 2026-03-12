import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('panda_nuts.csv')

# Basic check
print(df.head())
print(df.columns)
print(df.shape)

# Create efficiency

df['efficiency'] = df['nuts_opened'] / df['seconds']

# Clean: drop rows with missing values in relevant columns

relevant_cols = ['efficiency', 'age', 'sex', 'help', 'chimpanzee']
df_sub = df[relevant_cols].dropna()
print('Rows after dropna:', df_sub.shape)

# Encode categorical

df_sub['sex'] = df_sub['sex'].astype('category')
df_sub['help'] = df_sub['help'].astype('category')

# OLS with categorical predictors

model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df_sub).fit(cov_type='HC3')
print(model.summary())

# OLS with cluster-robust SE by chimpanzee

model_cluster = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df_sub).fit(
    cov_type='cluster', cov_kwds={'groups': df_sub['chimpanzee']}
)
print(model_cluster.summary())

# Mixed effects model (random intercept for chimpanzee)

try:
    md = smf.mixedlm('efficiency ~ age + C(sex) + C(help)', data=df_sub, groups=df_sub['chimpanzee'])
    mdf = md.fit(reml=False)
    print(mdf.summary())
except Exception as e:
    print('MixedLM failed:', e)

# group means

print('Efficiency by sex:')
print(df_sub.groupby('sex')['efficiency'].agg(['mean', 'median', 'count']))
print('Efficiency by help:')
print(df_sub.groupby('help')['efficiency'].agg(['mean', 'median', 'count']))
print('Correlation age-efficiency:', df_sub[['age', 'efficiency']].corr().iloc[0, 1])
