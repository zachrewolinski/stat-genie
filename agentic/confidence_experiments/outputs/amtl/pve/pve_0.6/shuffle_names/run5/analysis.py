import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

amtl = pd.read_csv('amtl.csv')

# Map columns
# outcome: genus (AMTL frequency measure)
# genus category: tooth_class
# tooth class: sockets
# age at death: pop
# sex probability: stdev_age

amtl['is_human'] = (amtl['tooth_class'] == 'Homo sapiens').astype(int)

model = smf.ols('genus ~ is_human + pop + stdev_age + C(sockets)', data=amtl).fit(cov_type='HC3')
print(model.summary())

# Also compare with categorical genus (tooth_class)
model2 = smf.ols('genus ~ C(tooth_class) + pop + stdev_age + C(sockets)', data=amtl).fit(cov_type='HC3')
print(model2.summary())

# Compute adjusted mean difference: predicted human vs non-human (average of non-human categories)
# We'll compute predictions at mean covariates and for each tooth class distribution (average over rows)
mean_pop = amtl['pop'].mean()
mean_sex = amtl['stdev_age'].mean()

# average over tooth class distribution
preds = {}
for sockets in amtl['sockets'].unique():
    for is_human in [0,1]:
        df = pd.DataFrame({'is_human':[is_human], 'pop':[mean_pop], 'stdev_age':[mean_sex], 'sockets':[sockets]})
        preds[(is_human, sockets)] = model.predict(df).iloc[0]

# weighted by sockets distribution
sockets_weights = amtl['sockets'].value_counts(normalize=True)
mean_pred_human = sum(preds[(1,s)] * sockets_weights[s] for s in sockets_weights.index)
mean_pred_nonhuman = sum(preds[(0,s)] * sockets_weights[s] for s in sockets_weights.index)

print('Adjusted mean (human):', mean_pred_human)
print('Adjusted mean (non-human):', mean_pred_nonhuman)
print('Difference:', mean_pred_human - mean_pred_nonhuman)

