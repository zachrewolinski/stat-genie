import json
import pandas as pd
import numpy as np
import statsmodels.api as sm


def main():
    df = pd.read_csv('hurricane.csv')
    # Focus on relevant columns
    # Outcome: fatalities (alldeaths). Use log1p to reduce skew and allow zeros.
    df = df.copy()
    df['log_deaths'] = np.log1p(df['alldeaths'])

    # Controls for storm severity and exposure proxies
    # Use category, min pressure, wind, normalized damage (ndam15), year.
    # Keep rows with non-missing required fields.
    cols = ['log_deaths', 'masfem', 'gender_mf', 'category', 'min', 'wind', 'ndam15', 'year']
    df_model = df[cols].dropna()

    results = {}

    # 1) Simple bivariate correlations
    corr_masfem = df_model[['log_deaths', 'masfem']].corr().iloc[0, 1]
    corr_gender = df_model[['log_deaths', 'gender_mf']].corr().iloc[0, 1]
    results['corr_masfem_log_deaths'] = float(corr_masfem)
    results['corr_gender_log_deaths'] = float(corr_gender)

    # 2) OLS with controls
    X = df_model[['masfem', 'category', 'min', 'wind', 'ndam15', 'year']]
    X = sm.add_constant(X)
    y = df_model['log_deaths']
    model = sm.OLS(y, X).fit(cov_type='HC3')
    results['ols_masfem'] = {
        'coef': float(model.params['masfem']),
        'pvalue': float(model.pvalues['masfem']),
        'stderr': float(model.bse['masfem']),
        'n': int(model.nobs),
        'r2': float(model.rsquared),
    }

    # 3) OLS with binary gender instead of masfem
    X2 = df_model[['gender_mf', 'category', 'min', 'wind', 'ndam15', 'year']]
    X2 = sm.add_constant(X2)
    model2 = sm.OLS(y, X2).fit(cov_type='HC3')
    results['ols_gender_mf'] = {
        'coef': float(model2.params['gender_mf']),
        'pvalue': float(model2.pvalues['gender_mf']),
        'stderr': float(model2.bse['gender_mf']),
        'n': int(model2.nobs),
        'r2': float(model2.rsquared),
    }

    # 4) Interaction: femininity x severity (category)
    df_model['masfem_x_cat'] = df_model['masfem'] * df_model['category']
    X3 = df_model[['masfem', 'category', 'masfem_x_cat', 'min', 'wind', 'ndam15', 'year']]
    X3 = sm.add_constant(X3)
    model3 = sm.OLS(y, X3).fit(cov_type='HC3')
    results['ols_masfem_x_cat'] = {
        'coef_masfem': float(model3.params['masfem']),
        'pvalue_masfem': float(model3.pvalues['masfem']),
        'coef_interaction': float(model3.params['masfem_x_cat']),
        'pvalue_interaction': float(model3.pvalues['masfem_x_cat']),
        'n': int(model3.nobs),
        'r2': float(model3.rsquared),
    }

    with open('analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
