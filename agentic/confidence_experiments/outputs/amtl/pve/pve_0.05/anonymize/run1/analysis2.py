import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('amtl.csv')

# Ensure categorical

df['feature8'] = df['feature8'].astype('category')
df['feature1'] = df['feature1'].astype('category')

# OLS with cluster-robust SE by specimen (feature2)
formula = "feature3 ~ C(feature8, Treatment(reference='Homo sapiens')) + C(feature1, Treatment(reference='Posterior')) + feature5 + feature7"
model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['feature2']})

print(model.summary())

# Extract comparisons Homo vs others
coef = model.params
pvals = model.pvalues

comparisons = {}
for genus in ['Pan', 'Papio', 'Pongo']:
    term = f"C(feature8, Treatment(reference='Homo sapiens'))[T.{genus}]"
    comparisons[genus] = {
        'diff_from_homo': coef[term],  # genus - homo
        'pvalue': pvals[term]
    }

print('\nComparisons (genus - Homo):')
for genus, stats in comparisons.items():
    print(genus, stats)

# Marginal means at average covariates for each genus
mean_age = df['feature5'].mean()
mean_sex = df['feature7'].mean()
# For tooth class, use observed distribution to compute average marginal effect
# Create design rows for each genus & tooth class, weighted by tooth class proportions
class_props = df['feature1'].value_counts(normalize=True)

rows = []
for genus in ['Homo sapiens', 'Pan', 'Papio', 'Pongo']:
    for tooth_class, weight in class_props.items():
        rows.append({
            'feature8': genus,
            'feature1': tooth_class,
            'feature5': mean_age,
            'feature7': mean_sex,
            'weight': weight
        })

pred_df = pd.DataFrame(rows)
# predictions
pred = model.get_prediction(pred_df).summary_frame()
# weighted average per genus
pred_df['pred'] = pred['mean']

marginal = pred_df.groupby('feature8').apply(lambda g: np.average(g['pred'], weights=g['weight']))
print('\nMarginal mean predicted feature3 (standardized AMTL):')
print(marginal)

