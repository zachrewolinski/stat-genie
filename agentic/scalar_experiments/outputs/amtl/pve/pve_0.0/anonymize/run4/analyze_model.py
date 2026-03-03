import pandas as pd
import statsmodels.formula.api as smf
import numpy as np


df = pd.read_csv('amtl.csv')

# Rename columns for clarity
col_map = {
    'feature1': 'tooth_class',
    'feature2': 'specimen_id',
    'feature3': 'amtl_missing',
    'feature4': 'observable_sockets',
    'feature5': 'age',
    'feature6': 'age_uncertainty',
    'feature7': 'sex',
    'feature8': 'genus',
    'feature9': 'region'
}

df = df.rename(columns=col_map)

# Create human indicator

df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Fit OLS model controlling for age, sex, tooth class
model = smf.ols('amtl_missing ~ is_human + age + sex + C(tooth_class)', data=df).fit(cov_type='HC3')

print(model.summary())

# Extract coefficient for is_human
coef = model.params['is_human']
se = model.bse['is_human']
pval = model.pvalues['is_human']

print('\nIs_human coef:', coef)
print('SE:', se)
print('p-value:', pval)

# Compute adjusted means for humans vs non-humans at average covariates and tooth class distribution
# Use model predictions with covariates at mean, and tooth class proportions

# Means
age_mean = df['age'].mean()
sex_mean = df['sex'].mean()

# Tooth class proportions
class_props = df['tooth_class'].value_counts(normalize=True)

# Build scenarios
preds = {}
for is_human in [0,1]:
    pred = 0.0
    for cls, prop in class_props.items():
        row = pd.DataFrame({
            'is_human': [is_human],
            'age': [age_mean],
            'sex': [sex_mean],
            'tooth_class': [cls]
        })
        pred += model.predict(row)[0] * prop
    preds[is_human] = pred

print('\nAdjusted mean amtl_missing (standardized) at average covariates:')
print(preds)
print('Difference (human - non-human):', preds[1]-preds[0])

