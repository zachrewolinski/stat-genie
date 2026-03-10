import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats


def main():
    df = pd.read_csv('hurricane.csv')

    # Identify key columns based on values and metadata
    # deaths: numeric small integers; femininity ratings: 1-11 scale (category, ind)
    # wind speed: year column (values ~75-190)
    # min pressure: ndam15 (values ~900-1000)
    # saffir-simpson category: gender_mf (1-5)
    # year of storm: wind (values 1950-2012)

    # Ensure numeric
    for col in ['name', 'category', 'ind', 'year', 'ndam15', 'gender_mf', 'wind']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Basic stats
    deaths = df['name']
    fem_index = df['category']
    fem_mturk = df['ind']

    # Correlations
    pearson_cat = stats.pearsonr(fem_index, deaths)
    spearman_cat = stats.spearmanr(fem_index, deaths)
    pearson_ind = stats.pearsonr(fem_mturk, deaths)
    spearman_ind = stats.spearmanr(fem_mturk, deaths)

    # Regression with log1p deaths
    df['log_deaths'] = np.log1p(df['name'])

    # Controls: wind speed, min pressure, category, year
    controls = ['year', 'ndam15', 'gender_mf', 'wind']

    # Model with femininity index (category)
    X1 = df[['category'] + controls].copy()
    X1 = sm.add_constant(X1)
    model1 = sm.OLS(df['log_deaths'], X1, missing='drop').fit()

    # Model with MTurk femininity rating (ind)
    X2 = df[['ind'] + controls].copy()
    X2 = sm.add_constant(X2)
    model2 = sm.OLS(df['log_deaths'], X2, missing='drop').fit()

    # Model with binary gender indicator masfem_mturk
    df['masfem_mturk'] = pd.to_numeric(df['masfem_mturk'], errors='coerce')
    X3 = df[['masfem_mturk'] + controls].copy()
    X3 = sm.add_constant(X3)
    model3 = sm.OLS(df['log_deaths'], X3, missing='drop').fit()

    # Interaction models: femininity * intensity (wind speed, category)
    df['cat_x_wind'] = df['category'] * df['year']
    df['cat_x_cat'] = df['category'] * df['gender_mf']
    X4 = df[['category', 'year', 'gender_mf', 'ndam15', 'wind', 'cat_x_wind', 'cat_x_cat']].copy()
    X4 = sm.add_constant(X4)
    model4 = sm.OLS(df['log_deaths'], X4, missing='drop').fit()

    df['ind_x_wind'] = df['ind'] * df['year']
    df['ind_x_cat'] = df['ind'] * df['gender_mf']
    X5 = df[['ind', 'year', 'gender_mf', 'ndam15', 'wind', 'ind_x_wind', 'ind_x_cat']].copy()
    X5 = sm.add_constant(X5)
    model5 = sm.OLS(df['log_deaths'], X5, missing='drop').fit()

    # Collect results
    results = {
        'n': int(df.shape[0]),
        'deaths_summary': {
            'min': float(deaths.min()),
            'max': float(deaths.max()),
            'mean': float(deaths.mean()),
            'median': float(deaths.median()),
        },
        'correlations': {
            'category_pearson_r': pearson_cat[0],
            'category_pearson_p': pearson_cat[1],
            'category_spearman_r': spearman_cat.correlation,
            'category_spearman_p': spearman_cat.pvalue,
            'ind_pearson_r': pearson_ind[0],
            'ind_pearson_p': pearson_ind[1],
            'ind_spearman_r': spearman_ind.correlation,
            'ind_spearman_p': spearman_ind.pvalue,
        },
        'model_category': {
            'coef': model1.params['category'],
            'pvalue': model1.pvalues['category'],
            'r2': model1.rsquared,
        },
        'model_ind': {
            'coef': model2.params['ind'],
            'pvalue': model2.pvalues['ind'],
            'r2': model2.rsquared,
        },
        'model_binary': {
            'coef': model3.params['masfem_mturk'],
            'pvalue': model3.pvalues['masfem_mturk'],
            'r2': model3.rsquared,
        },
        'model_category_interactions': {
            'coef_cat': model4.params['category'],
            'p_cat': model4.pvalues['category'],
            'coef_cat_x_wind': model4.params['cat_x_wind'],
            'p_cat_x_wind': model4.pvalues['cat_x_wind'],
            'coef_cat_x_cat': model4.params['cat_x_cat'],
            'p_cat_x_cat': model4.pvalues['cat_x_cat'],
            'r2': model4.rsquared,
        },
        'model_ind_interactions': {
            'coef_ind': model5.params['ind'],
            'p_ind': model5.pvalues['ind'],
            'coef_ind_x_wind': model5.params['ind_x_wind'],
            'p_ind_x_wind': model5.pvalues['ind_x_wind'],
            'coef_ind_x_cat': model5.params['ind_x_cat'],
            'p_ind_x_cat': model5.pvalues['ind_x_cat'],
            'r2': model5.rsquared,
        },
    }

    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
