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

    # OLS with controls
    ols = smf.ols(
        'affairs ~ C(children) + C(gender) + age + yearsmarried + '
        'religiousness + education + occupation + rating',
        data=df,
    ).fit()
    print('\nOLS with controls:')
    print(ols.summary().tables[1])

    # Logistic regression for any affair
    logit = smf.logit(
        'any_affair ~ C(children) + C(gender) + age + yearsmarried + '
        'religiousness + education + occupation + rating',
        data=df,
    ).fit(disp=0)
    print('\nLogit with controls:')
    print(logit.summary().tables[1])


if __name__ == '__main__':
    main()
