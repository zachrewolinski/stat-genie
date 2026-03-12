import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

print('Rows:', len(_df))
print('Columns:', _df.columns.tolist())

# Basic checks for num_amtl relative to sockets
num = _df['num_amtl']
sockets = _df['sockets']

print('num_amtl summary:', num.describe())
print('sockets summary:', sockets.describe())

# Check if num_amtl is integer-like
frac = np.abs(num - np.round(num))
print('num_amtl integer-like proportion (<=1e-6):', np.mean(frac <= 1e-6))
print('num_amtl in [0, sockets] proportion:', np.mean((num >= 0) & (num <= sockets)))

# Check if logistic transform might produce counts close to integers
p = 1 / (1 + np.exp(-num))
count_from_logit = p * sockets
frac2 = np.abs(count_from_logit - np.round(count_from_logit))
print('logit->count integer-like proportion (<=0.05):', np.mean(frac2 <= 0.05))
print('logit->count range:', count_from_logit.min(), count_from_logit.max())

# Prepare data for modeling
# We'll test two approaches:
# 1) Linear model on num_amtl
# 2) Binomial GLM on inferred counts if data looks like logit of proportion (fallback if plausible)

# Encode categorical variables
_df['genus'] = _df['genus'].astype('category')
_df['tooth_class'] = _df['tooth_class'].astype('category')

# Center age to improve stability
_df['age_c'] = _df['age'] - _df['age'].mean()

# Linear model
lm = smf.ols('num_amtl ~ C(genus) + age_c + prob_male + C(tooth_class)', data=_df).fit()
print(lm.summary())

# Extract human vs non-human contrast: we want if Homo sapiens has higher AMTL than others
# Use model's coefficient for genus categories relative to baseline.
# baseline is alphabetically first category in patsy; check categories
print('genus categories:', _df['genus'].cat.categories)

# We'll compute estimated marginal means for each genus at mean age/prob_male and average tooth_class
# Use model to predict
mean_age_c = 0.0
mean_prob_male = _df['prob_male'].mean()
# create grid
cats = _df['genus'].cat.categories
classes = _df['tooth_class'].cat.categories
rows = []
for g in cats:
    for tc in classes:
        rows.append({'genus': g, 'tooth_class': tc, 'age_c': mean_age_c, 'prob_male': mean_prob_male})

grid = pd.DataFrame(rows)
# predict
pred = lm.get_prediction(grid)
res = pred.summary_frame()
grid['pred'] = res['mean']

# average across tooth classes
avg = grid.groupby('genus')['pred'].mean()
print('Predicted mean num_amtl by genus (avg tooth_class):')
print(avg)

# Compute pairwise differences Homo sapiens vs others using linear contrasts
# We'll fit and use t_test for each comparison

# Build design rows for each genus, tooth_class averaged by taking mean across classes
# We'll compute differences by averaging design rows across tooth classes
from patsy import dmatrix

# design matrix for grid
X = dmatrix(lm.model.data.design_info.builder, grid, return_type='dataframe')
# average design rows per genus
X_avg = X.assign(genus=grid['genus']).groupby('genus').mean()

# contrast vectors
homo = X_avg.loc['Homo sapiens'].values

for g in cats:
    if g == 'Homo sapiens':
        continue
    other = X_avg.loc[g].values
    contrast = homo - other
    ttest = lm.t_test(contrast)
    effect = float(np.asarray(ttest.effect).ravel()[0])
    tval = float(np.asarray(ttest.tvalue).ravel()[0])
    pval = float(np.asarray(ttest.pvalue).ravel()[0])
    print(f'Contrast Homo sapiens - {g}: coef={effect:.4f}, t={tval:.3f}, p={pval:.4g}')

# Also test overall effect of genus
print('ANOVA for genus effect:')
try:
    anova = sm.stats.anova_lm(lm, typ=2)
    print(anova)
except Exception as e:
    print('ANOVA failed:', e)
