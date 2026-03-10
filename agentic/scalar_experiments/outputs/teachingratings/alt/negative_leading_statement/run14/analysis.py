import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats


def main():
    df = pd.read_csv('teachingratings.csv')

    # Basic cleaning: drop rows with missing values for variables of interest
    # (Dataset appears complete, but keep robust)
    cols_base = ['eval', 'beauty']
    df_base = df[cols_base].dropna()

    n = len(df_base)
    beauty = df_base['beauty']
    evals = df_base['eval']

    # Correlation
    r, p_corr = stats.pearsonr(beauty, evals)

    # Simple regression
    model_simple = smf.ols('eval ~ beauty', data=df).fit(cov_type='HC3')

    # Multivariate regression with typical controls
    # Avoid collinearity between students and allstudents by using both? We'll include both but results may be similar.
    # Include categorical variables as factors.
    formula = (
        'eval ~ beauty + age + C(gender) + C(minority) + C(native) + '
        'C(tenure) + C(division) + C(credits) + students + allstudents'
    )
    model_controls = smf.ols(formula, data=df).fit(cov_type='HC3')

    # Standardized effect of beauty from simple model
    beauty_sd = df['beauty'].std()
    eval_sd = df['eval'].std()
    coef_simple = model_simple.params['beauty']
    coef_controls = model_controls.params['beauty']

    # Effect per 1 SD beauty
    effect_1sd_simple = coef_simple * beauty_sd
    effect_1sd_controls = coef_controls * beauty_sd

    # Standardized beta
    beta_simple = coef_simple * beauty_sd / eval_sd
    beta_controls = coef_controls * beauty_sd / eval_sd

    results = {
        'n': n,
        'corr_r': r,
        'corr_p': p_corr,
        'simple_coef': coef_simple,
        'simple_p': model_simple.pvalues['beauty'],
        'controls_coef': coef_controls,
        'controls_p': model_controls.pvalues['beauty'],
        'beauty_sd': beauty_sd,
        'eval_sd': eval_sd,
        'effect_1sd_simple': effect_1sd_simple,
        'effect_1sd_controls': effect_1sd_controls,
        'beta_simple': beta_simple,
        'beta_controls': beta_controls,
        'simple_r2': model_simple.rsquared,
        'controls_r2': model_controls.rsquared,
    }

    with open('analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
