import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm

# Load
_df = pd.read_csv('amtl.csv')

# Clean
_df = _df[_df['sockets'] > 0].copy()

# Create human indicator
_df['is_human'] = (_df['genus'] == 'Homo sapiens').astype(int)
_df['prop_amtl'] = _df['num_amtl'] / _df['sockets']

# GLM with human indicator and controls
formula = 'prop_amtl ~ is_human + age + prob_male + C(tooth_class)'
model = smf.glm(formula=formula, data=_df, family=sm.families.Binomial(), freq_weights=_df['sockets'])
res = model.fit()

coef = res.params['is_human']
se = res.bse['is_human']
z = coef / se
p = 2 * (1 - norm.cdf(abs(z)))

# Predicted difference at mean covariates
mean_age = _df['age'].mean()
mean_prob_male = _df['prob_male'].mean()
mode_tooth = _df['tooth_class'].mode().iloc[0]

pred_df = pd.DataFrame([
    {'is_human': 1, 'age': mean_age, 'prob_male': mean_prob_male, 'tooth_class': mode_tooth},
    {'is_human': 0, 'age': mean_age, 'prob_male': mean_prob_male, 'tooth_class': mode_tooth},
])

preds = res.predict(pred_df)
diff = float(preds.iloc[0] - preds.iloc[1])

print('is_human coef:', coef)
print('SE:', se)
print('z:', z)
print('p:', p)
print('Predicted diff (human - nonhuman) at mean covariates:', diff)
print('\nModel summary:')
print(res.summary())
