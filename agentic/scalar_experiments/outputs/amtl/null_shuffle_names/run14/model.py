import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy


df = pd.read_csv('amtl.csv')

missing = df['genus'].astype(float)
total = df['age'].astype(float)
missing = np.minimum(missing, total)

mdf = pd.DataFrame({
    'missing': missing,
    'total': total,
    'genus_group': df['tooth_class'].astype('category'),
    'tooth_class': df['sockets'].astype('category'),
    'age_at_death': df['pop'].astype(float),
    'prob_male': df['stdev_age'].astype(float),
})

mdf = mdf[mdf['total'] > 0].copy()
mdf['genus_group'] = mdf['genus_group'].cat.reorder_categories(['Homo sapiens','Pan','Papio','Pongo'], ordered=False)

# design matrices
formula = 'missing_prop ~ C(genus_group) + C(tooth_class) + age_at_death + prob_male'
mdf['missing_prop'] = mdf['missing'] / mdf['total']

y, X = patsy.dmatrices(formula, data=mdf, return_type='dataframe')

model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=mdf['total'])
res = model.fit()
print(res.summary())

age_mean = mdf['age_at_death'].mean()
prob_mean = mdf['prob_male'].mean()

rows = []
for genus in ['Homo sapiens','Pan','Papio','Pongo']:
    for tc in mdf['tooth_class'].cat.categories:
        rows.append({'genus_group': genus, 'tooth_class': tc, 'age_at_death': age_mean, 'prob_male': prob_mean})

pred_df = pd.DataFrame(rows)

# use patsy build
pred_X = patsy.dmatrix('C(genus_group) + C(tooth_class) + age_at_death + prob_male', data=pred_df, return_type='dataframe')
pred = res.predict(pred_X)

pred_df['pred'] = pred
avg_pred = pred_df.groupby('genus_group')['pred'].mean()
print('\nAverage predicted AMTL proportion by genus_group:')
print(avg_pred)

nonhuman_mean = avg_pred[['Pan','Papio','Pongo']].mean()
print('\nHuman vs nonhuman mean difference:', avg_pred['Homo sapiens'] - nonhuman_mean)

params = res.params
conf = res.conf_int()

or_table = pd.DataFrame({
    'coef': params,
    'OR': np.exp(params),
    'OR_low': np.exp(conf[0]),
    'OR_high': np.exp(conf[1]),
})
print('\nOdds ratios:')
print(or_table)
