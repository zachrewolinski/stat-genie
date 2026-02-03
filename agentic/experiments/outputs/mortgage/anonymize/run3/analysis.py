import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'mortgage.csv'
df = pd.read_csv(path)

# Basic approval rates by gender
# feature2: 1 if female, 0 if male
# feature14: 1 if accepted, 0 if denied
approval_rates = df.groupby('feature2')['feature14'].mean()

# Prepare logistic regression
# Exclude feature11 (denial) because it's the complement of acceptance
# Exclude feature1 because it appears to be a row id
features = [
    'feature2',  # gender
    'feature3',
    'feature4',
    'feature5',
    'feature6',
    'feature7',
    'feature8',
    'feature9',
    'feature10',
    'feature12',
    'feature13'
]

df_model = df.replace([np.inf, -np.inf], np.nan)
df_model = df_model.dropna(subset=features + ['feature14'])

X = df_model[features]
X = sm.add_constant(X)
y = df_model['feature14']

logit = sm.Logit(y, X)
result = logit.fit(disp=False)

# Average marginal effect of gender on approval probability
X_female = X.copy()
X_male = X.copy()
X_female['feature2'] = 1
X_male['feature2'] = 0

pred_female = result.predict(X_female)
pred_male = result.predict(X_male)

ame_gender = (pred_female - pred_male).mean()

# Collect key outputs
output = {
    'approval_rate_male': float(approval_rates.loc[0.0]),
    'approval_rate_female': float(approval_rates.loc[1.0]),
    'logit_gender_coef': float(result.params['feature2']),
    'logit_gender_pvalue': float(result.pvalues['feature2']),
    'ame_gender': float(ame_gender)
}

print(output)
