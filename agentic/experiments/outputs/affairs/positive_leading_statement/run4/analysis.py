import pandas as pd
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv('affairs.csv')
    df['any_affair'] = (df['affairs'] > 0).astype(int)

    # Descriptive comparisons
    mean_by_children = df.groupby('children')['affairs'].mean()
    prop_any_by_children = df.groupby('children')['any_affair'].mean()

    print('Mean affairs by children:')
    print(mean_by_children)
    print('\nProportion with any affair by children:')
    print(prop_any_by_children)

    # Regression models with controls
    ols = smf.ols(
        'affairs ~ C(children) + age + yearsmarried + religiousness + education + occupation + rating + C(gender)',
        data=df,
    ).fit()
    logit = smf.logit(
        'any_affair ~ C(children) + age + yearsmarried + religiousness + education + occupation + rating + C(gender)',
        data=df,
    ).fit(disp=0)
    poisson = smf.poisson(
        'affairs ~ C(children) + age + yearsmarried + religiousness + education + occupation + rating + C(gender)',
        data=df,
    ).fit(disp=0)

    print('\nOLS coefficient for children=yes:')
    print(ols.params['C(children)[T.yes]'], 'p=', ols.pvalues['C(children)[T.yes]'])

    print('\nLogit coefficient for children=yes:')
    print(logit.params['C(children)[T.yes]'], 'p=', logit.pvalues['C(children)[T.yes]'])

    print('\nPoisson coefficient for children=yes:')
    print(poisson.params['C(children)[T.yes]'], 'p=', poisson.pvalues['C(children)[T.yes]'])


if __name__ == '__main__':
    main()
