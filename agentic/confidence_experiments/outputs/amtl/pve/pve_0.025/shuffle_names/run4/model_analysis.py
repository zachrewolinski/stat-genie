import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv('amtl.csv')
# Map columns to inferred meanings
# sockets -> tooth class (anterior/posterior/premolar)
# tooth_class -> genus (Homo sapiens, Pan, Papio, Pongo)
# stdev_age -> sex probability (0-1)
# pop -> estimated age at death

# Create binary indicator for human
DF = df.copy()
DF['is_human'] = (DF['tooth_class'] == 'Homo sapiens').astype(int)

# OLS with cluster-robust SE by specimen ID
model = smf.ols('genus ~ is_human + pop + stdev_age + C(sockets)', data=DF).fit(
    cov_type='cluster', cov_kwds={'groups': DF['prob_male']}
)

print(model.summary())

# Also fit with genus categories to compare directly
model2 = smf.ols('genus ~ C(tooth_class) + pop + stdev_age + C(sockets)', data=DF).fit(
    cov_type='cluster', cov_kwds={'groups': DF['prob_male']}
)
print('\nModel with genus categories:\n')
print(model2.summary())

# Compute adjusted mean difference (human vs non-human) from model2
# Use predicted values at mean covariates, average sockets distribution

# Prepare design for prediction: mean pop, mean stdev_age, average over sockets
mean_pop = DF['pop'].mean()
mean_sex = DF['stdev_age'].mean()

# compute predicted genus for each genus category averaged over sockets
preds = {}
for g in DF['tooth_class'].unique():
    sub = DF.copy()
    sub = sub.head(0)
    # construct rows for each socket category
    rows = []
    for s in DF['sockets'].unique():
        rows.append({'tooth_class': g, 'pop': mean_pop, 'stdev_age': mean_sex, 'sockets': s})
    pred = model2.predict(pd.DataFrame(rows)).mean()
    preds[g] = pred

print('\nAdjusted predictions (avg sockets, mean covariates):')
for k,v in preds.items():
    print(k, v)

# Compute difference human vs mean of non-human
non_human = [preds[g] for g in preds if g != 'Homo sapiens']
print('Human - mean(nonhuman):', preds['Homo sapiens'] - np.mean(non_human))

