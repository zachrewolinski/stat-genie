import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('affairs.csv')

# Interpret columns based on metadata mismatch
# 'religiousness' column appears to be children indicator (yes/no)
# 'age' column appears to be affairs frequency (0,1,2,3,7,12)

affairs = df['age']
children = df['religiousness']

# Basic checks
print('affairs unique:', sorted(affairs.unique()))
print('children counts:\n', children.value_counts())

# Any affair indicator
any_affair = (affairs > 0).astype(int)

# Group stats
grp = df.groupby(children)
mean_affairs = grp['age'].mean()
prop_any = grp.apply(lambda g: (g['age']>0).mean())

print('\nMean affairs by children:')
print(mean_affairs)
print('\nProportion any affair by children:')
print(prop_any)

# Difference in proportions (yes - no)
if 'yes' in prop_any.index and 'no' in prop_any.index:
    diff_prop = prop_any['yes'] - prop_any['no']
    print('\nDiff prop (yes - no):', diff_prop)

# Statistical tests
# Mann-Whitney U (non-param)
if set(children.unique()) >= {'yes','no'}:
    a_yes = affairs[children=='yes']
    a_no = affairs[children=='no']
    u_stat, p_u = stats.mannwhitneyu(a_yes, a_no, alternative='two-sided')
    print('\nMann-Whitney U p:', p_u)

    # t-test on affairs (unequal var)
    t_stat, p_t = stats.ttest_ind(a_yes, a_no, equal_var=False)
    print('t-test p:', p_t)

    # t-test on any affair
    t_stat2, p_t2 = stats.ttest_ind(any_affair[children=='yes'], any_affair[children=='no'], equal_var=False)
    print('t-test on any affair p:', p_t2)

    # effect size Cohen d for any affair
    mean_yes = any_affair[children=='yes'].mean()
    mean_no = any_affair[children=='no'].mean()
    n_yes = (children=='yes').sum()
    n_no = (children=='no').sum()
    var_yes = any_affair[children=='yes'].var(ddof=1)
    var_no = any_affair[children=='no'].var(ddof=1)
    pooled = ((n_yes-1)*var_yes + (n_no-1)*var_no) / (n_yes+n_no-2)
    d = (mean_yes - mean_no) / np.sqrt(pooled)
    print('Cohen d (any affair):', d)

    # Logistic regression (simple) using statsmodels
    try:
        import statsmodels.api as sm
        X = pd.get_dummies(children, drop_first=True)  # yes=1 if 'yes' is second? depends
        X = sm.add_constant(X)
        y = any_affair
        model = sm.Logit(y, X).fit(disp=False)
        print('\nLogit coef:')
        print(model.params)
        print('Logit p-values:')
        print(model.pvalues)
    except Exception as e:
        print('Logit failed:', e)
