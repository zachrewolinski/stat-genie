import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data

df = pd.read_csv('amtl.csv')

# Ensure categories
for col in ['genus', 'tooth_class']:
    df[col] = df[col].astype('category')

# Set reference category for genus and tooth_class (optional)
# We want Homo sapiens as reference
if 'Homo sapiens' in df['genus'].cat.categories:
    df['genus'] = df['genus'].cat.reorder_categories(
        ['Homo sapiens'] + [c for c in df['genus'].cat.categories if c != 'Homo sapiens'],
        ordered=False,
    )

# Build model formula
formula = 'num_amtl ~ C(genus) + age + prob_male + C(tooth_class)'

model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['specimen']})
print(model.summary())

# Extract genus coefficients (non-reference)
params = model.params
conf = model.conf_int()

# Compute pairwise difference: Homo sapiens vs each other genus
# With Homo as reference, coefficients for other genera represent (Other - Homo)
# So Homo higher if coefficients are negative.
results = []
for genus in df['genus'].cat.categories:
    if genus == 'Homo sapiens':
        continue
    coef = params.get(f'C(genus)[T.{genus}]', np.nan)
    ci_low, ci_high = conf.loc[f'C(genus)[T.{genus}]']
    pval = model.pvalues.get(f'C(genus)[T.{genus}]', np.nan)
    results.append((genus, coef, ci_low, ci_high, pval))

print('\nGenus contrasts (Other - Homo):')
for genus, coef, ci_low, ci_high, pval in results:
    print(f'{genus}: coef={coef:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}], p={pval:.4f}')

# Also compute predicted mean by genus at mean covariates for interpretability
mean_age = df['age'].mean()
mean_prob_male = df['prob_male'].mean()
# choose reference tooth_class as first category (Anterior) for comparability
# We'll compute across tooth_class by averaging predictions across categories equally

tooth_classes = df['tooth_class'].cat.categories

genera = df['genus'].cat.categories
preds = {}
for genus in genera:
    # average over tooth_class equally
    vals = []
    for tc in tooth_classes:
        row = pd.DataFrame({
            'genus': [genus],
            'age': [mean_age],
            'prob_male': [mean_prob_male],
            'tooth_class': [tc],
        })
        val = model.predict(row)[0]
        vals.append(val)
    preds[genus] = float(np.mean(vals))

print('\nPredicted mean num_amtl (avg over tooth_class, at mean age & prob_male):')
for genus, val in preds.items():
    print(f'{genus}: {val:.3f}')

