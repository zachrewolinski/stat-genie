import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


def main():
    df = pd.read_csv('affairs.csv')
    # children indicator: yes=1, no=0
    child_yes = df['feature6'].str.lower().eq('yes')

    # outcome: feature2 (affair frequency code)
    y = df['feature2'].astype(float)

    # group stats
    stats_tbl = df.groupby(child_yes)['feature2'].agg(['count', 'mean', 'median'])
    # proportion with any affair (>0)
    prop_any = df.assign(any_affair=y > 0).groupby(child_yes)['any_affair'].mean()

    # Welch t-test for mean difference
    y_yes = y[child_yes]
    y_no = y[~child_yes]
    t_res = stats.ttest_ind(y_yes, y_no, equal_var=False, nan_policy='omit')

    # Mann-Whitney U test for distribution shift
    try:
        mw_res = stats.mannwhitneyu(y_yes, y_no, alternative='two-sided')
    except ValueError:
        mw_res = None

    # logistic regression for any affair (controls for age, years married, relig, educ, occup, marriage rating, gender)
    X = df[['feature3','feature4','feature5','feature7','feature8','feature9','feature10']].copy()
    X['child_yes'] = child_yes.astype(int)
    # gender dummy
    X['male'] = (X['feature3'].str.lower() == 'male').astype(int)
    X = X.drop(columns=['feature3'])
    X = sm.add_constant(X, has_constant='add')
    y_bin = (y > 0).astype(int)
    logit = sm.Logit(y_bin, X)
    try:
        logit_res = logit.fit(disp=False)
    except Exception:
        logit_res = None

    # output results
    print('Group stats (children yes=True, no=False):')
    print(stats_tbl)
    print('\nProp any affair (>0):')
    print(prop_any)
    print('\nWelch t-test:')
    print(t_res)
    if mw_res is not None:
        print('\nMann-Whitney U:')
        print(mw_res)
    if logit_res is not None:
        print('\nLogit coef for child_yes:')
        print(logit_res.params['child_yes'])
        print('Logit p-value for child_yes:')
        print(logit_res.pvalues['child_yes'])

    # compute effect size (Cohen d)
    mean_yes = y_yes.mean()
    mean_no = y_no.mean()
    var_yes = y_yes.var(ddof=1)
    var_no = y_no.var(ddof=1)
    n_yes = y_yes.shape[0]
    n_no = y_no.shape[0]
    pooled_sd = np.sqrt(((n_yes-1)*var_yes + (n_no-1)*var_no) / (n_yes + n_no - 2))
    cohend = (mean_yes - mean_no) / pooled_sd if pooled_sd > 0 else np.nan
    print('\nCohen d (yes - no):', cohend)

    # also compute difference in proportion any affair
    diff_prop = prop_any.loc[True] - prop_any.loc[False]
    print('Diff prop any (yes - no):', diff_prop)


if __name__ == '__main__':
    main()
