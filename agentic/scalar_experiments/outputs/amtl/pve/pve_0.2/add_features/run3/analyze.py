import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np

# Load data

df = pd.read_csv('amtl.csv')

# Ensure categorical types
for col in ['genus','tooth_class']:
    df[col] = df[col].astype('category')

# Set reference category for genus as Homo sapiens
# Reorder categories to make Homo sapiens the reference
if 'Homo sapiens' in df['genus'].cat.categories:
    new_cats = ['Homo sapiens'] + [c for c in df['genus'].cat.categories if c != 'Homo sapiens']
    df['genus'] = df['genus'].cat.reorder_categories(new_cats, ordered=False)

# Fit OLS model with robust SE
formula = 'num_amtl ~ C(genus) + age + prob_male + C(tooth_class)'
model = smf.ols(formula, data=df).fit(cov_type='HC3')

print(model.summary())

# Extract coefficients for genus comparisons
params = model.params
pvalues = model.pvalues
conf_int = model.conf_int()

# coefficients for non-Homo genera (differences vs Homo)
results = []
for genus in ['Pan', 'Pongo', 'Papio']:
    term = f'C(genus)[T.{genus}]'
    if term in params:
        results.append({
            'genus': genus,
            'coef': params[term],
            'pvalue': pvalues[term],
            'ci_low': conf_int.loc[term,0],
            'ci_high': conf_int.loc[term,1]
        })

print('\nGenus differences vs Homo sapiens (negative means Homo higher):')
for r in results:
    print(r)

# Also compute joint F-test that all genus differences are 0
terms = [f'C(genus)[T.{g}]' for g in ['Pan','Pongo','Papio'] if f'C(genus)[T.{g}]' in params]
if terms:
    ftest = model.f_test(' = 0, '.join(terms) + ' = 0')
    print('\nJoint F-test for genus differences:', ftest)

# Compute adjusted mean for each genus at average age/prob_male and each tooth_class
avg_age = df['age'].mean()
avg_prob_male = df['prob_male'].mean()

pred_rows = []
for genus in df['genus'].cat.categories:
    for tooth in df['tooth_class'].cat.categories:
        pred_rows.append({'genus': genus, 'tooth_class': tooth, 'age': avg_age, 'prob_male': avg_prob_male})

pred_df = pd.DataFrame(pred_rows)

pred = model.get_prediction(pred_df)
summary = pred.summary_frame(alpha=0.05)

pred_df['pred'] = summary['mean']

# average across tooth classes (equal weight) for each genus
adj_means = pred_df.groupby('genus')['pred'].mean()
print('\nAdjusted mean num_amtl by genus (equal tooth_class weights):')
print(adj_means)

# Differences Homo vs others
if 'Homo sapiens' in adj_means.index:
    homo_mean = adj_means.loc['Homo sapiens']
    for genus in adj_means.index:
        if genus != 'Homo sapiens':
            print(f'Diff Homo - {genus}: {homo_mean - adj_means.loc[genus]:.4f}')
