import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats


def main():
    df = pd.read_csv('affairs.csv')

    # Binary indicator for any affair
    df['any_affair'] = (df['affairs'] > 0).astype(int)

    # Descriptive stats by children
    mean_affairs = df.groupby('children')['affairs'].mean()
    rate_any = df.groupby('children')['any_affair'].mean()

    # Welch t-test on affair counts by children
    no_affairs = df.loc[df['children'] == 'no', 'affairs']
    yes_affairs = df.loc[df['children'] == 'yes', 'affairs']
    ttest = stats.ttest_ind(no_affairs, yes_affairs, equal_var=False)

    # Logistic regression: any affair with controls
    logit = smf.logit(
        'any_affair ~ C(children) + age + yearsmarried + C(gender) + religiousness + education + occupation + rating',
        data=df
    ).fit(disp=0)

    # OLS as a robustness check on count outcome
    ols = smf.ols(
        'affairs ~ C(children) + age + yearsmarried + C(gender) + religiousness + education + occupation + rating',
        data=df
    ).fit()

    print('Mean affairs by children:')
    print(mean_affairs)
    print('\nAny-affair rate by children:')
    print(rate_any)
    print('\nWelch t-test (affairs ~ children):')
    print(ttest)
    print('\nLogit coefficients:')
    print(logit.params)
    print('\nLogit p-values:')
    print(logit.pvalues)
    print('\nOLS coefficients:')
    print(ols.params)
    print('\nOLS p-values:')
    print(ols.pvalues)


if __name__ == '__main__':
    main()
