import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv('hurricane.csv')

    # Basic checks
    df = df.copy()
    df['log_deaths'] = np.log1p(df['alldeaths'])

    # Simple correlations
    corr_masfem = df[['masfem', 'alldeaths', 'log_deaths']].corr()

    # OLS with controls for storm intensity
    # Controls: wind speed, minimum pressure, category, year
    formula_ols = 'log_deaths ~ masfem + wind + min + category + year'
    ols = smf.ols(formula=formula_ols, data=df).fit(cov_type='HC3')

    # Alternative using binary gender indicator
    formula_ols_gender = 'log_deaths ~ gender_mf + wind + min + category + year'
    ols_gender = smf.ols(formula=formula_ols_gender, data=df).fit(cov_type='HC3')

    # Poisson regression on deaths with same controls
    formula_pois = 'alldeaths ~ masfem + wind + min + category + year'
    pois = smf.glm(formula=formula_pois, data=df, family=sm.families.Poisson()).fit(cov_type='HC3')

    # Print results
    print('Rows:', len(df))
    print('\nCorrelation matrix (masfem, alldeaths, log_deaths):')
    print(corr_masfem)

    print('\nOLS (log1p deaths) with masfem + controls:')
    print(ols.summary())

    print('\nOLS (log1p deaths) with gender_mf + controls:')
    print(ols_gender.summary())

    print('\nPoisson (deaths) with masfem + controls:')
    print(pois.summary())

    # Extract key stats for convenience
    def coef_info(model, term):
        return {
            'coef': model.params.get(term, np.nan),
            'pval': model.pvalues.get(term, np.nan)
        }

    print('\nKey coefficients:')
    print('masfem (OLS):', coef_info(ols, 'masfem'))
    print('gender_mf (OLS):', coef_info(ols_gender, 'gender_mf'))
    print('masfem (Poisson):', coef_info(pois, 'masfem'))


if __name__ == '__main__':
    main()
