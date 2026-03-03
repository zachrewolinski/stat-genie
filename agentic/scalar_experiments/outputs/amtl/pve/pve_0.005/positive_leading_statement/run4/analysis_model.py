import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
path = 'amtl.csv'
df = pd.read_csv(path)

# Create indicator for human
# genus values include 'Homo sapiens'
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Fit OLS with cluster-robust SE by specimen
formula = 'num_amtl ~ is_human + age + prob_male + C(tooth_class) + sockets'
model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['specimen']})

print(model.summary())

# Extract coefficient for is_human
coef = model.params['is_human']
se = model.bse['is_human']
ci_low, ci_high = model.conf_int().loc['is_human']
pval = model.pvalues['is_human']
print('is_human coef', coef)
print('is_human se', se)
print('is_human pval', pval)
print('is_human CI', ci_low, ci_high)

# Fit model with genus categorical for adjusted means
model_genus = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class) + sockets', data=df).fit(cov_type='cluster', cov_kwds={'groups': df['specimen']})
print(model_genus.summary())

# Compute adjusted mean for each genus at mean covariates
cov_means = {
    'age': df['age'].mean(),
    'prob_male': df['prob_male'].mean(),
    'sockets': df['sockets'].mean(),
}
# Use all tooth classes equally for prediction; average over them
classes = df['tooth_class'].unique()

preds = []
for genus in df['genus'].unique():
    for tooth_class in classes:
        row = {
            'genus': genus,
            'tooth_class': tooth_class,
            **cov_means,
        }
        preds.append(row)

pred_df = pd.DataFrame(preds)
pred_df['pred'] = model_genus.predict(pred_df)

# Average predictions over tooth classes per genus
adj_means = pred_df.groupby('genus')['pred'].mean()
print('Adjusted means by genus:')
print(adj_means)

# Pairwise differences: Homo vs each other genus
homo_mean = adj_means.loc['Homo sapiens']
for genus in adj_means.index:
    if genus == 'Homo sapiens':
        continue
    print('Homo -', genus, homo_mean - adj_means.loc[genus])
