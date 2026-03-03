import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Basic checks for num_amtl
num_amtl = _df['num_amtl']
summary = {
    'num_amtl_mean': float(num_amtl.mean()),
    'num_amtl_std': float(num_amtl.std(ddof=1)),
    'num_amtl_min': float(num_amtl.min()),
    'num_amtl_max': float(num_amtl.max()),
}
print('summary', summary)

# Fit linear model with cluster-robust SE by specimen
formula = 'num_amtl ~ C(genus, Treatment(reference="Homo sapiens")) + age + prob_male + C(tooth_class)'
model = smf.ols(formula=formula, data=_df)
res = model.fit(cov_type='cluster', cov_kwds={'groups': _df['specimen']})

print('\nModel coefficients (cluster-robust):')
print(res.summary().tables[1])

# Extract genus coefficients and p-values
comparisons = {}
for genus in ['Pan', 'Papio', 'Pongo']:
    term = f'C(genus, Treatment(reference="Homo sapiens"))[T.{genus}]'
    if term in res.params.index:
        comparisons[genus] = {
            'coef': float(res.params[term]),
            'pvalue': float(res.pvalues[term]),
            'stderr': float(res.bse[term]),
        }

print('\nComparisons vs Homo sapiens:')
for g, vals in comparisons.items():
    print(g, vals)

# Estimated marginal means at average covariates per genus
mean_age = _df['age'].mean()
mean_prob_male = _df['prob_male'].mean()
classes = _df['tooth_class'].unique()

pred_rows = []
for genus in ['Homo sapiens', 'Pan', 'Papio', 'Pongo']:
    for tooth_class in classes:
        pred_rows.append({'genus': genus, 'age': mean_age, 'prob_male': mean_prob_male, 'tooth_class': tooth_class})

pred_df = pd.DataFrame(pred_rows)
preds = res.get_prediction(pred_df)

pred_df['pred'] = preds.predicted_mean
emm = pred_df.groupby('genus')['pred'].mean().sort_values(ascending=False)
print('\nEstimated marginal mean num_amtl (avg over tooth classes):')
print(emm)

# Compute difference Homo vs average of non-human genera
nonhuman = emm.loc[['Pan', 'Papio', 'Pongo']].mean()
print('\nHomo sapiens vs non-human average difference:', float(emm.loc['Homo sapiens'] - nonhuman))
