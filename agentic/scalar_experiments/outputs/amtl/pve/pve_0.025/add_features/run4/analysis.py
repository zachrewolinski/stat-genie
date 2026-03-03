import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
_df = pd.read_csv('amtl.csv')

# Keep relevant columns and drop missing
cols = ['num_amtl', 'genus', 'age', 'prob_male', 'tooth_class']
df = _df[cols].dropna()

# Ensure categories
for col in ['genus', 'tooth_class']:
    df[col] = df[col].astype('category')

# Set Homo sapiens as reference
if 'Homo sapiens' in df['genus'].cat.categories:
    df['genus'] = df['genus'].cat.reorder_categories(
        ['Homo sapiens'] + [c for c in df['genus'].cat.categories if c != 'Homo sapiens']
    )

formula = 'num_amtl ~ C(genus, Treatment(reference="Homo sapiens")) + age + prob_male + C(tooth_class)'
model = smf.ols(formula, data=df).fit(cov_type='HC3')

print(model.summary())

# Extract coefficients for genus comparisons
coef = model.params
pvals = model.pvalues

comparisons = {}
for term in coef.index:
    if term.startswith('C(genus'):
        comparisons[term] = {'coef': coef[term], 'p': pvals[term]}

print('\nGenus comparisons vs Homo sapiens (other - Homo):')
for term, stats in comparisons.items():
    print(f"{term}: coef={stats['coef']:.4f}, p={stats['p']:.4g}")

# Compute adjusted mean for each genus at mean age, mean prob_male, and marginalizing tooth_class by sample proportions
mean_age = df['age'].mean()
mean_prob = df['prob_male'].mean()

tooth_props = df['tooth_class'].value_counts(normalize=True)

# Build prediction data for each genus and tooth_class, then weighted average
pred_rows = []
for genus in df['genus'].cat.categories:
    for tooth, prop in tooth_props.items():
        pred_rows.append({'genus': genus, 'age': mean_age, 'prob_male': mean_prob, 'tooth_class': tooth, 'weight': prop})

pred_df = pd.DataFrame(pred_rows)

pred_df['pred'] = model.predict(pred_df)

adj_means = (
    pred_df.groupby('genus')
    .apply(lambda g: np.average(g['pred'], weights=g['weight']))
    .sort_values(ascending=False)
)

print('\nAdjusted means (higher = more AMTL):')
print(adj_means)

# Pairwise differences Homo vs each other genus using linear hypothesis
from statsmodels.stats.contrast import ContrastResults

# Build contrast for each other genus
# Parameter names
params = model.params.index.tolist()

# Helper to build contrast vector

def contrast_vector(term):
    v = np.zeros(len(params))
    v[params.index(term)] = 1.0
    return v

print('\nPairwise contrasts (other - Homo):')
for term in comparisons.keys():
    v = contrast_vector(term)
    t_test = model.t_test(v)
    diff = float(t_test.effect)
    p = float(t_test.pvalue)
    ci_low, ci_high = t_test.conf_int()[0]
    print(f"{term}: diff={diff:.4f}, 95% CI [{ci_low:.4f}, {ci_high:.4f}], p={p:.4g}")

