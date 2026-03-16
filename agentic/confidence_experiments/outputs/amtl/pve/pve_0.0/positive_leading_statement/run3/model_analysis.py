import pandas as pd
import statsmodels.formula.api as smf

# Load data
file_path = 'amtl.csv'
df = pd.read_csv(file_path)

# Create binary indicator for modern humans
# Use Homo sapiens (exact string in dataset) as modern humans

df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# OLS with cluster-robust SE by specimen to account for repeated measures
model = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class)', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['specimen']}
)

print('Binary human model:')
print(model.summary())

# Effect size and p-value
coef = model.params['is_human']
se = model.bse['is_human']
print('is_human coef:', coef)
print('is_human SE:', se)
print('is_human p-value:', model.pvalues['is_human'])

# Model with genus categories (human as baseline)
# Ensure baseline is Homo sapiens by reordering categories

df['genus_cat'] = pd.Categorical(df['genus'], categories=['Homo sapiens', 'Pan', 'Pongo', 'Papio'])
model_genus = smf.ols('num_amtl ~ C(genus_cat) + age + prob_male + C(tooth_class)', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['specimen']}
)

print('\nGenus categorical model:')
print(model_genus.summary())

# Extract coefficients for genera vs human
for genus in ['Pan', 'Pongo', 'Papio']:
    term = f'C(genus_cat)[T.{genus}]'
    if term in model_genus.params:
        print(genus, 'vs Homo sapiens coef:', model_genus.params[term], 'p-value:', model_genus.pvalues[term])

# Compute adjusted mean difference at mean covariates (for interpretability)
mean_age = df['age'].mean()
mean_prob_male = df['prob_male'].mean()

# Build data for prediction: human vs non-human (set tooth class to each and average)
import numpy as np

classes = df['tooth_class'].unique()

pred_rows = []
for is_human in [0,1]:
    for tc in classes:
        pred_rows.append({'is_human': is_human, 'age': mean_age, 'prob_male': mean_prob_male, 'tooth_class': tc})

pred_df = pd.DataFrame(pred_rows)

# Use model to predict
pred_df['pred'] = model.predict(pred_df)

# Average over tooth classes
mean_pred = pred_df.groupby('is_human')['pred'].mean()
print('\nAdjusted mean (standardized AMTL) by is_human:')
print(mean_pred)
print('Difference (human - non-human):', mean_pred[1] - mean_pred[0])

