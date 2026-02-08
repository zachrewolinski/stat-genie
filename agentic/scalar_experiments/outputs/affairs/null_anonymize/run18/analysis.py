import pandas as pd
import numpy as np
from scipy import stats


def cohen_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = a.size
    nb = b.size
    if na < 2 or nb < 2:
        return np.nan
    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    pooled = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    if pooled <= 0:
        return 0.0
    return (a.mean() - b.mean()) / np.sqrt(pooled)


def main():
    df = pd.read_csv('affairs.csv')
    # feature6: children yes/no, feature2: affair frequency category
    df = df.dropna(subset=['feature6', 'feature2'])
    yes = df[df['feature6'] == 'yes']['feature2']
    no = df[df['feature6'] == 'no']['feature2']

    # Any affair indicator
    yes_any = (yes > 0).mean()
    no_any = (no > 0).mean()

    # Means
    yes_mean = yes.mean()
    no_mean = no.mean()

    # Welch t-test on feature2
    t_res = stats.ttest_ind(yes, no, equal_var=False, nan_policy='omit')

    # Mann-Whitney U test
    try:
        u_res = stats.mannwhitneyu(yes, no, alternative='two-sided')
    except Exception:
        u_res = None

    d = cohen_d(yes, no)

    print('n_yes', yes.size)
    print('n_no', no.size)
    print('any_affair_yes', yes_any)
    print('any_affair_no', no_any)
    print('mean_yes', yes_mean)
    print('mean_no', no_mean)
    print('diff_mean_yes_minus_no', yes_mean - no_mean)
    print('cohen_d_yes_minus_no', d)
    print('t_stat', t_res.statistic)
    print('t_pvalue', t_res.pvalue)
    if u_res is not None:
        print('u_stat', u_res.statistic)
        print('u_pvalue', u_res.pvalue)

    # Also compute difference in any-affair rates and its standard error
    # for simple proportion z-test approximation
    p1 = yes_any
    p2 = no_any
    n1 = yes.size
    n2 = no.size
    se = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    z = (p1 - p2) / se if se > 0 else np.nan
    print('diff_any_yes_minus_no', p1 - p2)
    print('z_diff_any', z)


if __name__ == '__main__':
    main()
