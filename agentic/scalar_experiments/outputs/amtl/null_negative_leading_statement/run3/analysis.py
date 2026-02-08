import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Basic cleaning
_df = _df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus'])

# Binary indicator for human
_df['human'] = (_df['genus'] == 'Homo sapiens').astype(int)

# Proportion of AMTL with binomial weights
_df['amtl_prop'] = _df['num_amtl'] / _df['sockets']

# Fit binomial GLM with weights = sockets
formula = 'amtl_prop ~ human + age + prob_male + C(tooth_class)'
model = smf.glm(
    formula=formula,
    data=_df,
    family=sm.families.Binomial(),
    var_weights=_df['sockets'],
)
result = model.fit()

# Extract human effect
coef = result.params['human']
se = result.bse['human']
pval = result.pvalues['human']

# Odds ratio
or_human = float(np.exp(coef))

# Predict difference at mean covariates, reference tooth_class = Anterior
means = _df[['age', 'prob_male']].mean()

pred_base = pd.DataFrame({
    'human': [0, 1],
    'age': [means['age']] * 2,
    'prob_male': [means['prob_male']] * 2,
    'tooth_class': ['Anterior'] * 2,
})

pred = result.predict(pred_base)
diff = float(pred.iloc[1] - pred.iloc[0])

print('N', len(_df))
print('coef', coef)
print('se', se)
print('pval', pval)
print('odds_ratio', or_human)
print('pred_diff', diff)
