import json
import pandas as pd
import numpy as np
import statsmodels.api as sm


def main():
    df = pd.read_csv('crofoot.csv')
    cols = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other']
    df = df[cols].copy()
    df = df.dropna()

    # Derived predictors
    df['rel_size'] = df['n_focal'] - df['n_other']
    df['loc_adv'] = df['dist_other'] - df['dist_focal']  # positive = contest closer to focal

    # Standardize for effect sizes
    df['rel_size_z'] = (df['rel_size'] - df['rel_size'].mean()) / df['rel_size'].std(ddof=0)
    df['loc_adv_z'] = (df['loc_adv'] - df['loc_adv'].mean()) / df['loc_adv'].std(ddof=0)

    y = df['win']
    X = sm.add_constant(df[['rel_size', 'loc_adv']])

    # Use GLM Binomial for robustness
    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit()

    # Standardized model for comparable effects
    Xz = sm.add_constant(df[['rel_size_z', 'loc_adv_z']])
    model_z = sm.GLM(y, Xz, family=sm.families.Binomial())
    result_z = model_z.fit()

    def odds_ratio(coef):
        return float(np.exp(coef))

    summary = {
        'n': int(len(df)),
        'coef': result.params.to_dict(),
        'pvalues': result.pvalues.to_dict(),
        'odds_ratios': {k: odds_ratio(v) for k, v in result.params.items()},
        'coef_z': result_z.params.to_dict(),
        'pvalues_z': result_z.pvalues.to_dict(),
        'odds_ratios_z': {k: odds_ratio(v) for k, v in result_z.params.items()},
    }

    # Also fit single-predictor models for robustness
    single = {}
    for pred in ['rel_size', 'loc_adv']:
        X1 = sm.add_constant(df[[pred]])
        res1 = sm.GLM(y, X1, family=sm.families.Binomial()).fit()
        single[pred] = {
            'coef': res1.params.to_dict(),
            'pvalues': res1.pvalues.to_dict(),
            'odds_ratios': {k: odds_ratio(v) for k, v in res1.params.items()},
        }
    summary['single_predictor'] = single

    with open('analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
