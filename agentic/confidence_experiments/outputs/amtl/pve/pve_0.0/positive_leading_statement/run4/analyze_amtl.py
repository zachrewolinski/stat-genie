import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Load data
DF = pd.read_csv('amtl.csv')

# Create human vs non-human indicator
DF['is_human'] = (DF['genus'] == 'Homo sapiens').astype(int)

# Model 1: human vs non-human
m1 = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class)', data=DF).fit(cov_type='HC3')

# Model 2: genus categorical (Homo sapiens baseline)
m2 = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=DF).fit(cov_type='HC3')

# Extract stats for human effect
coef_human = m1.params['is_human']
se_human = m1.bse['is_human']
p_human = m1.pvalues['is_human']
ci_low, ci_high = m1.conf_int().loc['is_human'].tolist()

# Compute adjusted mean difference (human vs non-human) with average covariates
# For interpretability, use model to predict for each group at mean covariates and most common tooth_class
mean_age = DF['age'].mean()
mean_prob_male = DF['prob_male'].mean()
mode_tooth = DF['tooth_class'].mode().iloc[0]

pred_nonhuman = m1.predict(pd.DataFrame({
    'is_human': [0],
    'age': [mean_age],
    'prob_male': [mean_prob_male],
    'tooth_class': [mode_tooth],
}))

pred_human = m1.predict(pd.DataFrame({
    'is_human': [1],
    'age': [mean_age],
    'prob_male': [mean_prob_male],
    'tooth_class': [mode_tooth],
}))

# Genus-specific differences vs Homo sapiens
# Identify coefficients for non-human genera
nonhuman_coefs = {k: v for k, v in m2.params.items() if k.startswith('C(genus)')}
nonhuman_pvals = {k: v for k, v in m2.pvalues.items() if k.startswith('C(genus)')}

summary = {
    'n': len(DF),
    'coef_human': float(coef_human),
    'se_human': float(se_human),
    'p_human': float(p_human),
    'ci_low': float(ci_low),
    'ci_high': float(ci_high),
    'pred_nonhuman': float(pred_nonhuman.iloc[0]),
    'pred_human': float(pred_human.iloc[0]),
    'nonhuman_coefs': {k: float(v) for k, v in nonhuman_coefs.items()},
    'nonhuman_pvals': {k: float(v) for k, v in nonhuman_pvals.items()},
}

print(json.dumps(summary, indent=2))
