import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data
_df = pd.read_csv('amtl.csv')

# Keep only relevant columns
cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus']
_df = _df[cols].copy()

# Drop rows with missing values in relevant columns
_df = _df.dropna()

# Ensure categories
_df['genus'] = _df['genus'].astype('category')
_df['tooth_class'] = _df['tooth_class'].astype('category')

print('Rows:', len(_df))
print('Genus counts:')
print(_df['genus'].value_counts())
print('\nTooth class counts:')
print(_df['tooth_class'].value_counts())

# Use Homo sapiens as reference for genus
# Use Anterior as reference for tooth_class (default alphabetical; explicitly set for clarity)
_df['genus'] = _df['genus'].cat.reorder_categories(
    ['Homo sapiens', 'Pan', 'Papio', 'Pongo'],
    ordered=False
)
_df['tooth_class'] = _df['tooth_class'].cat.reorder_categories(
    ['Anterior', 'Posterior', 'Premolar'],
    ordered=False
)

# Fit OLS model
model = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=_df).fit()
print('\nOLS summary (key coefficients):')
print(model.summary().tables[1])

# Pairwise comparisons: Homo sapiens vs others
# In this parameterization, coefficients for C(genus)[T.Pan], etc represent (Pan - Homo).
# We test if Homo > other => coefficient < 0.

comparisons = ['C(genus)[T.Pan]', 'C(genus)[T.Papio]', 'C(genus)[T.Pongo]']
results = []
for term in comparisons:
    coef = model.params[term]
    se = model.bse[term]
    tval = model.tvalues[term]
    pval = model.pvalues[term]
    results.append((term, coef, se, tval, pval))

print('\nGenus comparisons vs Homo sapiens (other - Homo):')
for term, coef, se, tval, pval in results:
    print(f"{term}: coef={coef:.4f}, se={se:.4f}, t={tval:.3f}, p={pval:.4g}")

# Predicted means at average covariates for each genus (for interpretability)
mean_age = _df['age'].mean()
mean_prob_male = _df['prob_male'].mean()
# We'll set tooth_class to each class and average predictions across them equally.

genera = ['Homo sapiens', 'Pan', 'Papio', 'Pongo']
classes = ['Anterior', 'Posterior', 'Premolar']

preds = {}
for g in genera:
    vals = []
    for tc in classes:
        row = pd.DataFrame({
            'genus': [g],
            'age': [mean_age],
            'prob_male': [mean_prob_male],
            'tooth_class': [tc]
        })
        vals.append(model.predict(row).iloc[0])
    preds[g] = float(np.mean(vals))

print('\nPredicted mean num_amtl (avg over tooth classes, at mean age/sex):')
for g, val in preds.items():
    print(f"{g}: {val:.4f}")

# Also try adding sockets as covariate to check robustness
model2 = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class) + sockets', data=_df).fit()
print('\nOLS + sockets (key coefficients):')
print(model2.summary().tables[1])

results2 = []
for term in comparisons:
    coef = model2.params[term]
    se = model2.bse[term]
    tval = model2.tvalues[term]
    pval = model2.pvalues[term]
    results2.append((term, coef, se, tval, pval))

print('\nGenus comparisons vs Homo sapiens with sockets (other - Homo):')
for term, coef, se, tval, pval in results2:
    print(f"{term}: coef={coef:.4f}, se={se:.4f}, t={tval:.3f}, p={pval:.4g}")
