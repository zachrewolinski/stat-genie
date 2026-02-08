import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

# Load data
_df = pd.read_csv('affairs.csv')

df = _df.copy()

# Normalize children to categorical with 'no' as reference
if df['children'].dtype.name == 'category':
    df['children'] = df['children'].astype(str)

df['children'] = df['children'].str.lower().str.strip()

# Binary outcome: any affair

df['affair_any'] = (df['affairs'] > 0).astype(int)

# Basic group stats

group_stats = df.groupby('children').agg(
    n=('affairs','size'),
    mean_affairs=('affairs','mean'),
    median_affairs=('affairs','median'),
    prop_any=('affair_any','mean')
).reset_index()

# Difference in means (yes - no)
mean_yes = group_stats.loc[group_stats['children']=='yes','mean_affairs'].values[0]
mean_no = group_stats.loc[group_stats['children']=='no','mean_affairs'].values[0]
prop_yes = group_stats.loc[group_stats['children']=='yes','prop_any'].values[0]
prop_no = group_stats.loc[group_stats['children']=='no','prop_any'].values[0]

# Two-sample t-test (unequal var) for affairs counts

aff_yes = df.loc[df['children']=='yes','affairs']
aff_no = df.loc[df['children']=='no','affairs']

t_stat, t_p = stats.ttest_ind(aff_yes, aff_no, equal_var=False)

# Difference in proportions (any affair)

def prop_ztest(x1, n1, x2, n2):
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z = (p1 - p2) / se if se > 0 else np.nan
    p = 2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return z, p

x1 = df.loc[df['children']=='yes','affair_any'].sum()
N1 = df.loc[df['children']=='yes','affair_any'].count()
x2 = df.loc[df['children']=='no','affair_any'].sum()
N2 = df.loc[df['children']=='no','affair_any'].count()

z_stat, z_p = prop_ztest(x1, N1, x2, N2)

# Logistic regression for any affair
# Controls: gender, age, yearsmarried, religiousness, education, occupation, rating
logit_model = smf.logit(
    'affair_any ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating',
    data=df
).fit(disp=False)

# Negative binomial regression for count of affairs (handles overdispersion)
nb_model = smf.glm(
    'affairs ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating',
    data=df,
    family=sm.families.NegativeBinomial()
).fit()

# Extract effect for children
# In statsmodels, with C(children), coefficient for C(children)[T.yes] (yes vs no reference)
logit_coef = logit_model.params.get('C(children)[T.yes]', np.nan)
logit_p = logit_model.pvalues.get('C(children)[T.yes]', np.nan)

nb_coef = nb_model.params.get('C(children)[T.yes]', np.nan)
nb_p = nb_model.pvalues.get('C(children)[T.yes]', np.nan)

# Print results
print('GROUP_STATS')
print(group_stats)
print('MEAN_DIFF_yes_minus_no', mean_yes - mean_no)
print('PROP_DIFF_yes_minus_no', prop_yes - prop_no)
print('TTEST', t_stat, t_p)
print('PROP_ZTEST', z_stat, z_p)
print('LOGIT_COEF', logit_coef, 'P', logit_p)
print('NB_COEF', nb_coef, 'P', nb_p)

