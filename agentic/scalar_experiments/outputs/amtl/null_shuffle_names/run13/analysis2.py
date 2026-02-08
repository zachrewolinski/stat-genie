import pandas as pd
import numpy as np
import statsmodels.api as sm

path = 'amtl.csv'
df = pd.read_csv(path)

df = df.rename(columns={
    'sockets': 'tooth_class',
    'tooth_class': 'genus_cat',
    'genus': 'num_missing',
    'age': 'num_sockets',
    'pop': 'age_at_death',
    'stdev_age': 'prob_male',
    'prob_male': 'specimen_id'
})

mask = (df['num_missing'] >= 0) & (df['num_sockets'] > 0) & (df['num_missing'] <= df['num_sockets'])
clean = df.loc[mask].copy()

# Fix categories for consistent dummies
all_genus = sorted(clean['genus_cat'].unique())
all_tooth = sorted(clean['tooth_class'].unique())
clean['genus_cat'] = pd.Categorical(clean['genus_cat'], categories=all_genus)
clean['tooth_class'] = pd.Categorical(clean['tooth_class'], categories=all_tooth)

# Build design matrix manually
X = pd.get_dummies(clean[['genus_cat', 'tooth_class']], drop_first=True)
X = pd.concat([X, clean[['age_at_death', 'prob_male']]], axis=1)
X = sm.add_constant(X, has_constant='add')
X = X.astype(float)

y = (clean['num_missing'] / clean['num_sockets']).astype(float)

model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=clean['num_sockets']).fit()
print(model.summary())

# Predicted marginal rates for each genus
pred_rates = {}
for genus in all_genus:
    tmp = clean.copy()
    tmp['genus_cat'] = pd.Categorical([genus] * len(tmp), categories=all_genus)
    tmp['tooth_class'] = pd.Categorical(tmp['tooth_class'], categories=all_tooth)
    X_tmp = pd.get_dummies(tmp[['genus_cat', 'tooth_class']], drop_first=True)
    X_tmp = pd.concat([X_tmp, tmp[['age_at_death', 'prob_male']]], axis=1)
    X_tmp = X_tmp.reindex(columns=X.columns.drop('const'), fill_value=0)
    X_tmp = sm.add_constant(X_tmp, has_constant='add')
    X_tmp = X_tmp.astype(float)
    pred = model.predict(X_tmp)
    pred_rates[genus] = float(np.mean(pred))

print('pred_rates', pred_rates)

human = pred_rates.get('Homo sapiens')
non_humans = [v for k, v in pred_rates.items() if k != 'Homo sapiens']
nonhuman_mean = float(np.mean(non_humans))
print('human', human, 'nonhuman_mean', nonhuman_mean, 'diff', human - nonhuman_mean)

for k, v in pred_rates.items():
    if k == 'Homo sapiens':
        continue
    print('diff vs', k, human - v)

print('params')
print(model.params)
