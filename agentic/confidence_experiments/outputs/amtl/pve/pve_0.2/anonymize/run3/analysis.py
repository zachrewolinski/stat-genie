import pandas as pd
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('amtl.csv')

# Create indicator for Homo sapiens vs non-human primates

df['is_human'] = (df['feature8'] == 'Homo sapiens').astype(int)

# OLS model with controls: age (feature5), sex (feature7), tooth class (feature1)
model = smf.ols('feature3 ~ is_human + feature5 + feature7 + C(feature1)', data=df).fit(cov_type='HC3')
print(model.summary())

# Extract coefficient and p-value for is_human
coef = model.params['is_human']
pval = model.pvalues['is_human']

print('\ncoef_is_human', coef)
print('pval_is_human', pval)

# Calculate adjusted means for human vs non-human at average covariates
mean_age = df['feature5'].mean()
mean_sex = df['feature7'].mean()

# Most common tooth class as reference
ref_class = df['feature1'].value_counts().idxmax()

# Predictions for each class with mean covariates, then average across class distribution
classes = df['feature1'].unique()
class_probs = df['feature1'].value_counts(normalize=True)

preds = {}
for human in [0,1]:
    pred_sum = 0.0
    for cls in classes:
        row = {'is_human': human, 'feature5': mean_age, 'feature7': mean_sex, 'feature1': cls}
        pred = model.predict(pd.DataFrame([row]))[0]
        pred_sum += pred * class_probs[cls]
    preds[human] = pred_sum

print('adjusted_mean_nonhuman', preds[0])
print('adjusted_mean_human', preds[1])
print('adjusted_mean_diff', preds[1]-preds[0])
