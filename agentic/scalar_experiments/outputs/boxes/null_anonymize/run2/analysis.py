import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data

df = pd.read_csv('boxes.csv')

# Outcome definitions
# feature1: 1=unchosen option, 2=majority option, 3=minority option

df['majority'] = (df['feature1'] == 2).astype(int)
df['social_reliance'] = df['feature1'].isin([2, 3]).astype(int)

# Prepare categorical site

df['site'] = df['feature5'].astype('category')

# --- Social reliance analysis ---
# Chi-square for site differences
site_social_table = pd.crosstab(df['site'], df['social_reliance'])
chi2_social_site = stats.chi2_contingency(site_social_table)

# Logistic regression for age effect on social reliance
model_social_age = smf.glm('social_reliance ~ feature3', data=df, family=sm.families.Binomial()).fit()

# --- Majority preference analysis ---
# Chi-square for site differences
site_majority_table = pd.crosstab(df['site'], df['majority'])
chi2_majority_site = stats.chi2_contingency(site_majority_table)

# Logistic regression for age effect on majority preference
model_majority_age = smf.glm('majority ~ feature3', data=df, family=sm.families.Binomial()).fit()

# Additional: majority preference conditional on demonstrated choices
# if child chose demonstrated option, what fraction chose majority?

cond_df = df[df['social_reliance'] == 1].copy()
cond_df['majority_given_demo'] = (cond_df['feature1'] == 2).astype(int)

cond_site_majority_table = pd.crosstab(cond_df['site'], cond_df['majority_given_demo'])
chi2_majority_cond_site = stats.chi2_contingency(cond_site_majority_table)

model_majority_cond_age = smf.glm('majority_given_demo ~ feature3', data=cond_df, family=sm.families.Binomial()).fit()

# Summaries
summary = {
    'n': len(df),
    'social_reliance_overall': df['social_reliance'].mean(),
    'majority_overall': df['majority'].mean(),
    'majority_given_demo_overall': cond_df['majority_given_demo'].mean(),
    'chi2_social_site_p': chi2_social_site[1],
    'social_age_p': model_social_age.pvalues['feature3'],
    'chi2_majority_site_p': chi2_majority_site[1],
    'majority_age_p': model_majority_age.pvalues['feature3'],
    'chi2_majority_cond_site_p': chi2_majority_cond_site[1],
    'majority_cond_age_p': model_majority_cond_age.pvalues['feature3'],
    'social_age_coef': model_social_age.params['feature3'],
    'majority_age_coef': model_majority_age.params['feature3'],
    'majority_cond_age_coef': model_majority_cond_age.params['feature3'],
}

# Compute per-site rates for context
site_rates = df.groupby('site').agg(
    social_reliance_rate=('social_reliance', 'mean'),
    majority_rate=('majority', 'mean')
).reset_index()

cond_site_rates = cond_df.groupby('site').agg(
    majority_given_demo_rate=('majority_given_demo', 'mean')
).reset_index()

site_rates = site_rates.merge(cond_site_rates, on='site', how='left')

# Decide scalar based on strength of evidence
# We treat strong evidence if both site and age effects are significant (p<0.05)
# for either social reliance or majority preference.

pvals = {
    'social_site': summary['chi2_social_site_p'],
    'social_age': summary['social_age_p'],
    'majority_site': summary['chi2_majority_site_p'],
    'majority_age': summary['majority_age_p'],
    'majority_cond_site': summary['chi2_majority_cond_site_p'],
    'majority_cond_age': summary['majority_cond_age_p'],
}

# Count significant results
sig_count = sum(p < 0.05 for p in pvals.values())

# Base score by significant count and effect sizes
score = 0

# Emphasize evidence for variation across cultures (site) and developmental stages (age)
site_sig = (pvals['social_site'] < 0.05) or (pvals['majority_site'] < 0.05) or (pvals['majority_cond_site'] < 0.05)
age_sig = (pvals['social_age'] < 0.05) or (pvals['majority_age'] < 0.05) or (pvals['majority_cond_age'] < 0.05)

if site_sig and age_sig:
    score = 70
elif site_sig or age_sig:
    score = 40
else:
    score = 0

# Adjust based on number of significant tests
score += (sig_count - 2) * 5

# Clamp to [-100, 100]
score = int(max(-100, min(100, round(score))))

print('SUMMARY')
print(summary)
print('\nPVALUES', pvals)
print('\nSITE RATES')
print(site_rates)
print('\nSCORE', score)

# Save score
with open('conclusion.txt', 'w') as f:
    f.write(str(score))
