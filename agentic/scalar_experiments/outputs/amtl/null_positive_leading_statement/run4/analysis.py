import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

# Load data
_df = pd.read_csv('amtl.csv')

# Basic cleaning: drop rows with missing key fields
required_cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus']
df = _df.dropna(subset=required_cols).copy()

# Create binary indicator for humans vs non-human primates
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Filter invalid rows for binomial modeling
df = df[(df['sockets'] > 0) & (df['num_amtl'] >= 0) & (df['num_amtl'] <= df['sockets'])].copy()

# Ensure categorical tooth_class
# (statsmodels formula will treat object/category as categorical)

# Fit binomial GLM using successes/failures
df['num_fail'] = df['sockets'] - df['num_amtl']
endog = df[['num_amtl', 'num_fail']].to_numpy()
exog = patsy.dmatrix(
    'is_human + age + prob_male + C(tooth_class)',
    data=df,
    return_type='dataframe'
)
design_info = exog.design_info
model = sm.GLM(endog, exog, family=sm.families.Binomial()).fit()

# Extract effect for is_human
coef = model.params['is_human']
se = model.bse['is_human']
pval = model.pvalues['is_human']

# Odds ratio
or_val = float(np.exp(coef))

# Summaries
print('n_rows_used', len(df))
print('coef_is_human', coef)
print('se_is_human', se)
print('pval_is_human', pval)
print('odds_ratio_is_human', or_val)

# Also compute predicted mean AMTL rate at average covariates for human vs non-human
avg_age = df['age'].mean()
avg_male = df['prob_male'].mean()
# Use most common tooth_class for baseline
baseline_tooth = df['tooth_class'].mode().iloc[0]

pred_df = pd.DataFrame({
    'is_human': [0, 1],
    'age': [avg_age, avg_age],
    'prob_male': [avg_male, avg_male],
    'tooth_class': [baseline_tooth, baseline_tooth],
})

pred_exog = patsy.build_design_matrices([design_info], pred_df, return_type='dataframe')[0]
pred = model.predict(pred_exog)
print('baseline_tooth_class', baseline_tooth)
print('pred_rate_nonhuman', pred.iloc[0])
print('pred_rate_human', pred.iloc[1])
