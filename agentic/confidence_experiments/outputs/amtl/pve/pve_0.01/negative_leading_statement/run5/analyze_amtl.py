import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


df = pd.read_csv('amtl.csv')

# Set category ordering to make Homo sapiens reference
# Use pandas C() with Treatment; set reference by reordering categories

df['genus'] = pd.Categorical(df['genus'], categories=[
    'Homo sapiens', 'Pan', 'Pongo', 'Papio'
])

df['tooth_class'] = pd.Categorical(df['tooth_class'])

# Fit OLS model (num_amtl appears continuous)
model = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=df).fit()

# ANOVA for genus effect
anova_results = anova_lm(model, typ=2)

print('OLS summary:')
print(model.summary())
print('\nANOVA (Type II):')
print(anova_results)

# Extract coefficients for genus comparisons vs Homo sapiens
coef = model.params
pvals = model.pvalues

comparisons = {}
for g in ['Pan', 'Pongo', 'Papio']:
    term = f'C(genus)[T.{g}]'
    if term in coef:
        comparisons[g] = {
            'coef_vs_homo': coef[term],
            'pval': pvals[term]
        }

print('\nGenus comparisons vs Homo sapiens:')
for g, info in comparisons.items():
    print(g, info)

# Compute adjusted means for each genus at mean age/prob_male and common tooth_class distribution
# We'll predict at average covariates and average over tooth_class distribution to get adjusted mean.
mean_age = df['age'].mean()
mean_prob_male = df['prob_male'].mean()

# Use empirical tooth_class distribution for weighting
class_probs = df['tooth_class'].value_counts(normalize=True)

pred_rows = []
for g in df['genus'].cat.categories:
    for tc, w in class_probs.items():
        pred_rows.append({'genus': g, 'age': mean_age, 'prob_male': mean_prob_male, 'tooth_class': tc, 'weight': w})

pred_df = pd.DataFrame(pred_rows)
pred_df['pred'] = model.predict(pred_df)

adj_means = pred_df.groupby('genus').apply(lambda d: np.average(d['pred'], weights=d['weight']))

print('\nAdjusted means (weighted by tooth_class, at mean age/prob_male):')
print(adj_means)

# Differences Homo - others
homo_mean = adj_means.loc['Homo sapiens']
print('\nAdjusted mean differences (Homo - others):')
for g in ['Pan', 'Pongo', 'Papio']:
    if g in adj_means.index:
        print(g, homo_mean - adj_means.loc[g])

