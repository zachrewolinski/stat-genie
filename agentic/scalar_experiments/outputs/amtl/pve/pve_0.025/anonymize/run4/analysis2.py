import pandas as pd
import statsmodels.formula.api as smf
import numpy as np


df = pd.read_csv('amtl.csv')

# Create Homo indicator

df['homo'] = (df['feature8'] == 'Homo sapiens').astype(int)

# OLS with covariates
model = smf.ols('feature3 ~ homo + feature5 + feature7 + C(feature1) + feature4', data=df).fit(cov_type='HC3')
print(model.summary())

# Model with genus categorical
model2 = smf.ols('feature3 ~ C(feature8) + feature5 + feature7 + C(feature1) + feature4', data=df).fit(cov_type='HC3')
print(model2.summary())

# Pairwise comparisons Homo vs each non-human (difference in means adjusted)
# compute using model2: predicted for each genus at mean covariates
cov_means = {
    'feature5': df['feature5'].mean(),
    'feature7': df['feature7'].mean(),
    'feature4': df['feature4'].mean(),
    'feature1': df['feature1'].mode()[0]
}

def predict_for(genus):
    row = cov_means.copy()
    row['feature8'] = genus
    return model2.predict(pd.DataFrame([row]))[0]

for g in df['feature8'].unique():
    print(g, predict_for(g))

# simple difference between homo and non-human in residualized sense
# t-test of homo coefficient already in model1

