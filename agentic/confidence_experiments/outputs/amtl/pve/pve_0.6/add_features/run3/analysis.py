import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data
df = pd.read_csv('amtl.csv')

# Keep relevant columns
cols = ['num_amtl','age','prob_male','tooth_class','genus']

# Drop missing
df = df[cols].dropna().copy()

# Create human indicator
df['human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Ensure categorical
df['tooth_class'] = df['tooth_class'].astype('category')
df['genus'] = df['genus'].astype('category')

# Model: human vs non-human, controlling for age, sex (prob_male), tooth_class
model = smf.ols('num_amtl ~ human + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')

# Also model with genus categories for pairwise comparisons
model_genus = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')

# Compute mean difference human vs nonhuman adjusted? We'll use coefficient from model.
human_coef = model.params['human']
human_se = model.bse['human']
human_p = model.pvalues['human']

# For effect size, compute standardized since num_amtl already standardized; report coefficient in SD units.

# Compute sample sizes per genus
genus_counts = df['genus'].value_counts().to_dict()

# For pairwise comparison: Homo sapiens vs each other genus
# We'll compute contrasts using model_genus

# Prepare results
results = {
    'n': int(df.shape[0]),
    'genus_counts': genus_counts,
    'human_coef': human_coef,
    'human_se': human_se,
    'human_p': human_p,
    'model_r2': model.rsquared,
    'model_adj_r2': model.rsquared_adj,
}

# Pairwise differences relative to baseline in model_genus
# Statsmodels uses first category alphabetically as baseline; we'll compute predicted mean for each genus at mean covariates.

# Build design matrix for each genus at mean age/prob_male and reference tooth_class (most common)
mean_age = df['age'].mean()
mean_prob_male = df['prob_male'].mean()
# choose most common tooth_class
mode_tooth = df['tooth_class'].mode().iloc[0]

# Predict for each genus
preds = {}
for g in df['genus'].cat.categories:
    row = pd.DataFrame({
        'genus':[g],
        'age':[mean_age],
        'prob_male':[mean_prob_male],
        'tooth_class':[mode_tooth]
    })
    preds[g] = float(model_genus.predict(row)[0])

results['preds_at_means'] = preds

# Print results in a stable format
print('N', results['n'])
print('genus_counts', results['genus_counts'])
print('human_coef', human_coef)
print('human_se', human_se)
print('human_p', human_p)
print('model_r2', results['model_r2'])
print('model_adj_r2', results['model_adj_r2'])
print('preds_at_means', results['preds_at_means'])
