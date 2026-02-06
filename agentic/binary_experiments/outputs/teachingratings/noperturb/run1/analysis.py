import pandas as pd
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv('teachingratings.csv')

    # OLS with common controls; use robust SEs (HC3) for heteroskedasticity.
    formula = (
        'eval ~ beauty + age + C(gender) + C(minority) + C(credits) + '
        'C(division) + C(native) + C(tenure) + students + allstudents'
    )
    model = smf.ols(formula, data=df).fit(cov_type='HC3')

    # Bivariate model as a simple check.
    model_bi = smf.ols('eval ~ beauty', data=df).fit(cov_type='HC3')

    print('Multivariate OLS (HC3) coefficient on beauty:')
    print('coef =', round(model.params['beauty'], 4),
          'SE =', round(model.bse['beauty'], 4),
          'p =', round(model.pvalues['beauty'], 6))

    print('\nBivariate OLS (HC3) coefficient on beauty:')
    print('coef =', round(model_bi.params['beauty'], 4),
          'SE =', round(model_bi.bse['beauty'], 4),
          'p =', round(model_bi.pvalues['beauty'], 6))


if __name__ == '__main__':
    main()
