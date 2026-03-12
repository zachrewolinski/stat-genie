import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data
path = 'amtl.csv'
df = pd.read_csv(path)

# Basic cleaning
# Ensure categorical types for genus and tooth_class
for col in ['genus', 'tooth_class']:
    df[col] = df[col].astype('category')

# Binary indicator for human vs non-human
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Fit OLS with robust SE
model = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')

# Fit OLS with genus categories to compare specific genera
model_genus = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')

# Extract coefficient for is_human
coef = model.params['is_human']
se = model.bse['is_human']
pt = model.pvalues['is_human']

# Compute adjusted means for human vs non-human at average covariates
# Use model predictions at mean covariates
mean_age = df['age'].mean()
mean_prob_male = df['prob_male'].mean()
# Reference tooth_class is the first alphabetical category; set to 'Anterior' explicitly for interpretability
ref_tooth = 'Anterior'

pred_human = model.predict(pd.DataFrame({
    'is_human':[1],
    'age':[mean_age],
    'prob_male':[mean_prob_male],
    'tooth_class':[ref_tooth]
}))[0]
pred_nonhuman = model.predict(pd.DataFrame({
    'is_human':[0],
    'age':[mean_age],
    'prob_male':[mean_prob_male],
    'tooth_class':[ref_tooth]
}))[0]

# Compare Homo sapiens to each non-human genus with contrasts from model_genus
# Compute adjusted difference: Homo minus each genus at mean covariates/tooth class
# Use prediction for each genus

def pred_for_genus(genus):
    return model_genus.predict(pd.DataFrame({
        'genus':[genus],
        'age':[mean_age],
        'prob_male':[mean_prob_male],
        'tooth_class':[ref_tooth]
    }))[0]

preds = {g: pred_for_genus(g) for g in df['genus'].cat.categories}

# Build contrasts using model_genus coefficients
# We'll compute pairwise differences Homo vs each genus using model_genus and robust covariance
from patsy import build_design_matrices

# Construct design matrix for each genus + mean covariates
def design_row(genus):
    d = pd.DataFrame({
        'genus':[genus],
        'age':[mean_age],
        'prob_male':[mean_prob_male],
        'tooth_class':[ref_tooth]
    })
    return build_design_matrices([model_genus.model.data.design_info], d)[0]

# Compute contrasts Homo vs other genera
contrast_results = {}
X_homo = design_row('Homo sapiens')
for g in df['genus'].cat.categories:
    if g == 'Homo sapiens':
        continue
    X_g = design_row(g)
    # contrast vector
    c = (X_homo - X_g)
    # t-test for contrast
    t_res = model_genus.t_test(c)
    contrast_results[g] = {
        'diff': float(preds['Homo sapiens'] - preds[g]),
        'pvalue': float(t_res.pvalue),
        'tvalue': float(t_res.tvalue)
    }

print('OLS is_human coef', coef)
print('OLS is_human SE', se)
print('OLS is_human p', pt)
print('Pred human', pred_human, 'Pred nonhuman', pred_nonhuman)
print('Contrast results', contrast_results)
