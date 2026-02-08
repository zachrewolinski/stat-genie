import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('amtl.csv')

# Basic cleaning: drop rows with missing key fields
cols = ['num_amtl','sockets','age','prob_male','tooth_class','genus']
# Coerce
for c in ['num_amtl','sockets','age','prob_male']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Keep positive sockets
df = df.dropna(subset=cols)
df = df[df['sockets'] > 0]
df = df[(df['num_amtl'] >= 0) & (df['num_amtl'] <= df['sockets'])]

# Create proportion
# Use GLM binomial with weights

# Set categorical
for c in ['tooth_class','genus']:
    df[c] = df[c].astype('category')

# Set reference: non-human? We'll set genus with Homo sapiens as reference? Actually want compare Homo vs others.
# Use treatment coding with non-human as reference; so create binary indicator for Homo.

# Create binary indicator for Homo sapiens
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Also include genus for non-humans? We'll use is_human + genus as factor? That would be collinear.
# We'll use is_human only to compare humans vs non-humans.

# model with age, prob_male, tooth_class
# Use binomial counts to avoid invalid proportion issues
df['num_present'] = df['sockets'] - df['num_amtl']

formula = 'num_amtl + num_present ~ is_human + age + prob_male + C(tooth_class)'
model = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
res = model.fit()

print(res.summary())

# Extract coefficient and p-value for is_human
coef = res.params['is_human']
se = res.bse['is_human']
pval = res.pvalues['is_human']

# Compute odds ratio
import numpy as np
or_val = np.exp(coef)

print('coef_is_human', coef)
print('se_is_human', se)
print('pval_is_human', pval)
print('odds_ratio', or_val)

# Also compute predicted probabilities at mean covariates for human vs non
mean_age = df['age'].mean()
mean_prob_male = df['prob_male'].mean()

# For each tooth_class baseline? We'll compute average over tooth_class distribution
# Compute predicted probability for each tooth_class, weighted by frequency
weights = df['tooth_class'].value_counts(normalize=True)

def predict_prob(is_human):
    probs = []
    for tc, w in weights.items():
        row = pd.DataFrame({'is_human':[is_human], 'age':[mean_age], 'prob_male':[mean_prob_male], 'tooth_class':[tc]})
        p = res.predict(row)[0]
        probs.append(p * w)
    return sum(probs)

p_non = predict_prob(0)
p_hum = predict_prob(1)
print('pred_prob_nonhuman', p_non)
print('pred_prob_human', p_hum)
print('diff', p_hum - p_non)
