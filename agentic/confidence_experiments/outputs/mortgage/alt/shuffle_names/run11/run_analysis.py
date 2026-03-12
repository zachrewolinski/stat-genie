import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats


df = pd.read_csv('mortgage.csv')

id_col = 'bad_history'
approval_col = 'deny'
gender_col = 'denied_PMI'

# Basic rates
approval_rate = df[approval_col].mean()
rate_female = df.loc[df[gender_col]==1, approval_col].mean()
rate_male = df.loc[df[gender_col]==0, approval_col].mean()

n_female = (df[gender_col]==1).sum()
n_male = (df[gender_col]==0).sum()

cont = pd.crosstab(df[gender_col], df[approval_col])
chi2, p_chi, _, _ = stats.chi2_contingency(cont)

exclude = {id_col, approval_col, 'self_employed'}
controls = [c for c in df.columns if c not in exclude]

X = df[[gender_col] + [c for c in controls if c != gender_col]].copy()
X = sm.add_constant(X, has_constant='add')

y = df[approval_col]

# Drop missing
data = pd.concat([y, X], axis=1).dropna()
y_clean = data[approval_col]
X_clean = data.drop(columns=[approval_col])

logit_model = sm.Logit(y_clean, X_clean).fit(disp=False)
coef_gender = logit_model.params[gender_col]
pval_gender = logit_model.pvalues[gender_col]
odds_ratio = np.exp(coef_gender)

print('approval_rate', approval_rate)
print('rate_female', rate_female, 'rate_male', rate_male, 'diff', rate_female - rate_male)
print('n_female', n_female, 'n_male', n_male)
print('chi2', chi2, 'p', p_chi)
print('logit coef', coef_gender, 'odds_ratio', odds_ratio, 'p', pval_gender)
