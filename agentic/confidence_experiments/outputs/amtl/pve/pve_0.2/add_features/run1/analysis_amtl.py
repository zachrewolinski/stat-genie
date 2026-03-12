import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Variables of interest
cols = ['num_amtl', 'age', 'prob_male', 'tooth_class', 'genus']

df = _df[cols].copy()

# Drop missing
_df_len = len(df)
df = df.dropna()

# Human indicator
_df_human = df['genus'].astype(str).str.strip()

# Mark human
human_label = 'Homo sapiens'
df['is_human'] = (_df_human == human_label).astype(int)

# OLS with robust SEs
model = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')

coef = model.params['is_human']
se = model.bse['is_human']
pval = model.pvalues['is_human']

# Effect size in SD units
outcome_sd = df['num_amtl'].std(ddof=1)
std_effect = coef / outcome_sd if outcome_sd != 0 else np.nan

# Predicted marginal means at average covariates for human vs non-human
mean_age = df['age'].mean()
mean_prob_male = df['prob_male'].mean()

# Use most common tooth_class as reference level for prediction
ref_tooth = df['tooth_class'].mode().iloc[0]

pred_df = pd.DataFrame({
    'is_human': [0, 1],
    'age': [mean_age, mean_age],
    'prob_male': [mean_prob_male, mean_prob_male],
    'tooth_class': [ref_tooth, ref_tooth],
})

pred = model.predict(pred_df)

# Save results for downstream use
results = {
    'n_total': int(len(df)),
    'n_human': int(df['is_human'].sum()),
    'n_nonhuman': int((df['is_human'] == 0).sum()),
    'coef_is_human': float(coef),
    'se_is_human': float(se),
    'pval_is_human': float(pval),
    'std_effect': float(std_effect),
    'pred_nonhuman': float(pred.iloc[0]),
    'pred_human': float(pred.iloc[1]),
    'ref_tooth': str(ref_tooth),
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
