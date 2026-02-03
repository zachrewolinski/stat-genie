import pandas as pd
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv('affairs.csv')

    # Basic group comparison
    group_means = df.groupby('children')['affairs'].mean()
    group_any = (df.assign(any_affair=df['affairs'] > 0)
                   .groupby('children')['any_affair'].mean())

    df['children_yes'] = (df['children'] == 'yes').astype(int)

    # Unadjusted OLS
    ols_simple = smf.ols('affairs ~ children_yes', data=df).fit()

    # Adjusted OLS with common covariates
    ols_adjusted = smf.ols(
        'affairs ~ children_yes + gender + age + yearsmarried + '
        'religiousness + education + occupation + rating',
        data=df
    ).fit()

    # Logistic regression for any affair
    df['any_affair'] = (df['affairs'] > 0).astype(int)
    logit = smf.logit(
        'any_affair ~ children_yes + gender + age + yearsmarried + '
        'religiousness + education + occupation + rating',
        data=df
    ).fit(disp=False)

    print('Group mean affairs by children:')
    print(group_means)
    print('\nShare with any affair by children:')
    print(group_any)
    print('\nUnadjusted OLS (affairs ~ children):')
    print(ols_simple.summary())
    print('\nAdjusted OLS (with covariates):')
    print(ols_adjusted.summary())
    print('\nLogit for any affair (with covariates):')
    print(logit.summary())


if __name__ == '__main__':
    main()
