import json
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data

df = pd.read_csv('amtl.csv')

# Ensure categories
for col in ['genus', 'tooth_class', 'specimen']:
    df[col] = df[col].astype('category')

# Use Homo sapiens as reference category for genus
# statsmodels allows setting category order
if 'Homo sapiens' in df['genus'].cat.categories:
    cats = list(df['genus'].cat.categories)
    # put Homo sapiens first for reference
    cats = ['Homo sapiens'] + [c for c in cats if c != 'Homo sapiens']
    df['genus'] = df['genus'].cat.reorder_categories(cats, ordered=False)

# Fit OLS model with clustered SE by specimen to account for repeated measures
formula = 'num_amtl ~ C(genus) + age + prob_male + C(tooth_class)'
model = smf.ols(formula, data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['specimen']}
)

# Extract coefficients for genus differences (others vs Homo)
coef = model.params
pvals = model.pvalues

# Build results for each non-human genus
results = {}
for g in df['genus'].cat.categories:
    if g == 'Homo sapiens':
        continue
    term = f'C(genus)[T.{g}]'
    if term in coef:
        results[g] = {
            'coef_vs_homo': float(coef[term]),
            'p_value': float(pvals[term])
        }

# Calculate adjusted mean for each genus at mean covariates for context
mean_age = df['age'].mean()
mean_prob_male = df['prob_male'].mean()
# Use most frequent tooth_class as reference? We'll compute marginal mean across tooth_class proportions.
# We'll compute predicted mean per genus averaging over tooth_class distribution.

tooth_dist = df['tooth_class'].value_counts(normalize=True)

def pred_for_genus(genus):
    # average across tooth_class distribution
    preds = []
    for tc, w in tooth_dist.items():
        row = pd.DataFrame({
            'genus': [genus],
            'age': [mean_age],
            'prob_male': [mean_prob_male],
            'tooth_class': [tc]
        })
        preds.append(float(model.predict(row)) * w)
    return sum(preds)

adj_means = {g: pred_for_genus(g) for g in df['genus'].cat.categories}

# Also compute overall genus effect (F-test)
# Compare model with and without genus
model_nogenus = smf.ols('num_amtl ~ age + prob_male + C(tooth_class)', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['specimen']}
)

# Use Wald test for genus terms
# Create restriction matrix for genus terms
terms = [f'C(genus)[T.{g}]' for g in df['genus'].cat.categories if g != 'Homo sapiens']

wald = model.wald_test(terms)

out = {
    'n': int(df.shape[0]),
    'n_specimen': int(df['specimen'].nunique()),
    'results_vs_homo': results,
    'adj_means': {k: float(v) for k,v in adj_means.items()},
    'genus_wald_stat': float(wald.statistic),
    'genus_wald_p': float(wald.pvalue)
}

print(json.dumps(out, indent=2))
