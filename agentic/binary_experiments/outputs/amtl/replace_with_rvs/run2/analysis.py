import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Keep needed columns and drop missing
cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus']
df = _df[cols].dropna().copy()

# Derived variables
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Guard against non-positive sockets
if (df['sockets'] <= 0).any():
    raise ValueError('Found non-positive sockets, cannot model binomial counts.')

# Proportion response with binomial weights
# GLM: endog as proportion, freq_weights as trials
df = df[df['num_amtl'] <= df['sockets']].copy()
df['amtl_rate'] = df['num_amtl'] / df['sockets']
if (df['amtl_rate'] < 0).any() or (df['amtl_rate'] > 1).any():
    raise ValueError('Found amtl_rate outside [0,1] after filtering.')
formula = 'amtl_rate ~ is_human + age + prob_male + C(tooth_class)'
model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df['sockets']
)
result = model.fit(cov_type='HC1')

# Extract effect for human indicator
coef = result.params['is_human']
se = result.bse['is_human']
pval = result.pvalues['is_human']

# Predicted rates for human vs nonhuman at mean covariates
mean_age = df['age'].mean()
mean_prob_male = df['prob_male'].mean()
# Use most common tooth_class as baseline for interpretability
baseline_tooth = df['tooth_class'].value_counts().idxmax()

pred_df = pd.DataFrame({
    'is_human': [0, 1],
    'age': [mean_age, mean_age],
    'prob_male': [mean_prob_male, mean_prob_male],
    'tooth_class': [baseline_tooth, baseline_tooth]
})

pred = result.predict(pred_df)

# Summarize
summary = {
    'n_rows_used': int(df.shape[0]),
    'baseline_tooth_class': baseline_tooth,
    'coef_is_human_logit': float(coef),
    'se_is_human': float(se),
    'pval_is_human': float(pval),
    'pred_rate_nonhuman': float(pred.iloc[0]),
    'pred_rate_human': float(pred.iloc[1])
}

print('GLM binomial with logit link: num_amtl / sockets ~ is_human + age + prob_male + C(tooth_class)')
print(result.summary())
print('\nSummary:')
for k, v in summary.items():
    print(f'{k}: {v}')
