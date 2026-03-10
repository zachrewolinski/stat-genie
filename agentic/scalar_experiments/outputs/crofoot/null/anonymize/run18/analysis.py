import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
csv_path = 'crofoot.csv'
df = pd.read_csv(csv_path)

# Rename for clarity
# feature4: outcome (1 focal wins)
# feature7: focal group size, feature8: other group size
# feature5: distance of focal from its home center
# feature6: distance of other from its home center

# Construct predictors
# Relative group size: focal - other (positive means focal larger)
df['rel_group_size'] = df['feature7'] - df['feature8']
# Relative location: focal distance - other distance (positive means focal farther from its center)
df['rel_location'] = df['feature5'] - df['feature6']

# Outcome
y = df['feature4']

# Logistic regression with both predictors
X = df[['rel_group_size','rel_location']]
X = sm.add_constant(X)
model = sm.Logit(y, X)
res = model.fit(disp=False)

print('Logit results (rel_group_size, rel_location)')
print(res.summary())

# Also check simple models separately
for col in ['rel_group_size','rel_location']:
    X1 = sm.add_constant(df[[col]])
    m1 = sm.Logit(y, X1).fit(disp=False)
    print('\nSingle-predictor Logit:', col)
    print(m1.summary())

# Nonparametric: compare win rates by sign of rel predictors
for col in ['rel_group_size','rel_location']:
    groups = df.groupby(np.sign(df[col]))['feature4']
    # sign -1,0,1
    print(f"\nWin rate by sign of {col}:")
    for sign, vals in groups:
        win_rate = vals.mean()
        print(sign, win_rate, len(vals))

# Mann-Whitney / t-test on predictors by outcome
for col in ['rel_group_size','rel_location']:
    g1 = df[df['feature4']==1][col]
    g0 = df[df['feature4']==0][col]
    # t-test
    tstat, pval = stats.ttest_ind(g1, g0, equal_var=False)
    # Mann-Whitney
    ustat, pval_u = stats.mannwhitneyu(g1, g0, alternative='two-sided')
    print(f"\n{col}: t-test p={pval:.4f}, MW p={pval_u:.4f}")
    print(f"means: win={g1.mean():.3f} lose={g0.mean():.3f}")

# Save some key stats for later use
print('\nPseudo R2 (McFadden):', 1 - res.llf/res.llnull)
print('N:', len(df))
