import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('amtl.csv')

# Create binary human indicator

df['is_human'] = (df['feature8'] == 'Homo sapiens').astype(int)

# Outcome: missing teeth / observable sockets
# Use GLM binomial with weights = feature4

# Center/scale age maybe? We'll include raw age, sex, tooth class categorical

formula = 'prop_missing ~ is_human + feature5 + feature7 + C(feature1)'

df['prop_missing'] = df['feature3'] / df['feature4']

model = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), freq_weights=df['feature4']).fit()
print(model.summary())

# Extract human effect
coef = model.params['is_human']
se = model.bse['is_human']
from math import exp
odds_ratio = exp(coef)

# Compute predicted probabilities at mean age/sex and each tooth class? We'll use marginal effect.
# Average marginal effect of is_human
marginal = model.get_margeff(at='overall', method='dydx')
print(marginal.summary())

# For simple: compute predicted probability for human vs non-human at mean covariates and each tooth class proportionally
mean_age = df['feature5'].mean()
mean_sex = df['feature7'].mean()

# Build average predictions across tooth classes weighted by freq
weights = df['feature1'].value_counts(normalize=True)

def pred_prob(is_human, tooth):
    row = pd.DataFrame({'is_human':[is_human], 'feature5':[mean_age], 'feature7':[mean_sex], 'feature1':[tooth]})
    return model.predict(row)[0]

preds = {}
for tooth in weights.index:
    preds[tooth] = {
        'human': pred_prob(1, tooth),
        'nonhuman': pred_prob(0, tooth),
    }

avg_human = sum(preds[t]['human'] * weights[t] for t in weights.index)
avg_non = sum(preds[t]['nonhuman'] * weights[t] for t in weights.index)

print('is_human coef', coef, 'se', se, 'odds_ratio', odds_ratio)
print('avg_pred_human', avg_human, 'avg_pred_non', avg_non, 'diff', avg_human-avg_non)

# Also test model with full genus categorical
formula2 = 'prop_missing ~ C(feature8) + feature5 + feature7 + C(feature1)'
model2 = smf.glm(formula=formula2, data=df, family=sm.families.Binomial(), freq_weights=df['feature4']).fit()
print(model2.summary())

# Compare Homo vs others with contrasts from model2
# Baseline is first alphabetical? We'll print params
print('params2', model2.params)

