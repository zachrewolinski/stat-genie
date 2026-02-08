import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Clean / derive
# Ensure no negative or missing sockets
_df = _df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus'])
_df = _df[_df['sockets'] >= _df['num_amtl']]

# Model: binomial counts
# Use Homo sapiens as reference
formula = 'num_amtl + 0'

# Build endog as success/failure for GLM
endog = np.column_stack([_df['num_amtl'].astype(int), (_df['sockets'] - _df['num_amtl']).astype(int)])

# Design matrix via patsy with reference levels
# Use Treatment coding with Homo sapiens as reference for genus
# Use Treatment coding with Anterior as reference for tooth_class
import patsy

# Set categorical with references
_df['genus'] = _df['genus'].astype('category')
_df['tooth_class'] = _df['tooth_class'].astype('category')

rhs = 'C(genus, Treatment(reference="Homo sapiens")) + age + prob_male + C(tooth_class, Treatment(reference="Anterior"))'

# Build design matrix
y, X = patsy.dmatrices(
    f'num_amtl ~ {rhs}',
    data=_df,
    return_type='dataframe'
)

# GLM with binomial counts
model = sm.GLM(endog, X, family=sm.families.Binomial())
res = model.fit()

# Extract coefficients for genus
coef = res.params
se = res.bse
pvalues = res.pvalues

# Predicted probabilities by genus at mean covariates and average tooth class distribution
# Build a small prediction grid using mean age and prob_male and each genus and each tooth_class
mean_age = _df['age'].mean()
mean_prob_male = _df['prob_male'].mean()

genera = sorted(_df['genus'].unique())
tclasses = sorted(_df['tooth_class'].unique())

rows = []
for g in genera:
    for tc in tclasses:
        rows.append({'genus': g, 'age': mean_age, 'prob_male': mean_prob_male, 'tooth_class': tc})

pred_df = pd.DataFrame(rows)

# Build design for predictions
Xp = patsy.dmatrix(
    rhs,
    data=pred_df,
    return_type='dataframe'
)

pred_lin = np.dot(Xp, res.params)
pred_prob = 1 / (1 + np.exp(-pred_lin))
pred_df['pred_prob'] = pred_prob

# Average predicted probability per genus (averaging tooth classes equally)
mean_pred_by_genus = pred_df.groupby('genus')['pred_prob'].mean().sort_values(ascending=False)

# Likelihood ratio test for genus overall
# Fit reduced model without genus
X_reduced = patsy.dmatrix(
    'age + prob_male + C(tooth_class, Treatment(reference="Anterior"))',
    data=_df,
    return_type='dataframe'
)
model_reduced = sm.GLM(endog, X_reduced, family=sm.families.Binomial())
res_reduced = model_reduced.fit()

lr_stat = 2 * (res.llf - res_reduced.llf)
df_diff = res.df_model - res_reduced.df_model
from scipy import stats
lr_p = stats.chi2.sf(lr_stat, df_diff)

# Summarize
print('N rows:', len(_df))
print('Genus counts:', _df['genus'].value_counts().to_dict())
print('\nModel summary (genus coefficients):')
for name in coef.index:
    if name.startswith('C(genus'):
        print(name, 'coef', coef[name], 'se', se[name], 'p', pvalues[name])

print('\nMean predicted AMTL probability by genus (mean age/prob_male, equal tooth classes):')
print(mean_pred_by_genus)

print('\nLR test for genus overall: stat', lr_stat, 'df', df_diff, 'p', lr_p)

# Also compute effect: Homo vs non-human (average non-human predicted probs)
if 'Homo sapiens' in mean_pred_by_genus.index:
    homo = mean_pred_by_genus.loc['Homo sapiens']
    nonhuman = mean_pred_by_genus.drop('Homo sapiens').mean()
    diff = homo - nonhuman
    print('\nHomo predicted prob:', homo)
    print('Nonhuman avg predicted prob:', nonhuman)
    print('Difference (Homo - nonhuman):', diff)
