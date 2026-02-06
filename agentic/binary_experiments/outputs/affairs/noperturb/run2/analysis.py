import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttest_ind


def main():
    df = pd.read_csv('affairs.csv')
    df['affair_any'] = (df['affairs'] > 0).astype(int)
    df['children_yes'] = (df['children'] == 'yes').astype(int)

    group = df.groupby('children')
    summary = group['affairs'].agg(['count', 'mean', 'median']).rename(columns={'count': 'n'})
    prop_any = group['affair_any'].mean().rename('prop_any')
    summary = summary.join(prop_any)

    affairs_yes = df.loc[df['children'] == 'yes', 'affairs']
    affairs_no = df.loc[df['children'] == 'no', 'affairs']
    t_stat, t_p, _ = ttest_ind(affairs_yes, affairs_no, usevar='unequal')

    logit = smf.logit(
        'affair_any ~ children_yes + age + yearsmarried + religiousness + education + occupation + rating + C(gender)',
        data=df,
    ).fit(disp=False)

    ols = smf.ols(
        'affairs ~ children_yes + age + yearsmarried + religiousness + education + occupation + rating + C(gender)',
        data=df,
    ).fit()

    print('Summary by children')
    print(summary)
    print('\nT-test on affairs counts (children yes vs no):')
    print(f't={t_stat:.3f}, p={t_p:.4f}')

    print('\nLogit (any affair) coefficient for children_yes:')
    print(logit.params['children_yes'], 'p=', logit.pvalues['children_yes'])

    print('\nOLS (affair count) coefficient for children_yes:')
    print(ols.params['children_yes'], 'p=', ols.pvalues['children_yes'])

    summary.to_csv('children_summary.csv')


if __name__ == '__main__':
    main()
