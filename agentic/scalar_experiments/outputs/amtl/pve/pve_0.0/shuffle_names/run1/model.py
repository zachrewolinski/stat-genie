import pandas as pd
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('amtl.csv')

# Map columns based on metadata interpretation
# outcome: genus (AMTL measure)
# genus category: tooth_class
# tooth class: sockets
# age at death: pop
# sex: stdev_age
# specimen id: prob_male

# create human indicator

df['human'] = (df['tooth_class'] == 'Homo sapiens').astype(int)

# Fit OLS with cluster-robust SE by specimen
formula = 'genus ~ human + pop + stdev_age + C(sockets)'
model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['prob_male']})
print(model.summary())

# Also model with full genus categories for comparison
model2 = smf.ols('genus ~ C(tooth_class) + pop + stdev_age + C(sockets)', data=df).fit(cov_type='cluster', cov_kwds={'groups': df['prob_male']})
print('\nModel with genus categories:')
print(model2.summary())

# get estimated marginal means for each genus from model2
# use baseline and contrast
import numpy as np

cats = df['tooth_class'].unique()

# Use average values for covariates and reference category for sockets
mean_pop = df['pop'].mean()
mean_sex = df['stdev_age'].mean()

# choose reference sockets as Anterior if present
ref_socket = df['sockets'].unique()[0]

# build design and compute predicted means
preds = {}
for g in sorted(cats):
    # create row
    row = pd.DataFrame({
        'tooth_class':[g],
        'pop':[mean_pop],
        'stdev_age':[mean_sex],
        'sockets':[ref_socket]
    })
    pred = model2.predict(row)[0]
    preds[g] = pred

print('\nPredicted genus (AMTL measure) by genus at mean covariates:')
for g, p in preds.items():
    print(g, p)

