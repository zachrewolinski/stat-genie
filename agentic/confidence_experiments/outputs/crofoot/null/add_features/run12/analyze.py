import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def main():
    df = pd.read_csv('crofoot.csv')

    # Derived predictors
    df = df.copy()
    df['rel_size'] = df['n_focal'] - df['n_other']
    # Positive value means contest is closer to focal home range center than to other group
    df['loc_adv'] = df['dist_other'] - df['dist_focal']

    # Standardize predictors for interpretability (per SD)
    for col in ['rel_size', 'loc_adv']:
        df[col + '_z'] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

    y = df['win']
    X = df[['rel_size_z', 'loc_adv_z']]
    X = sm.add_constant(X)

    model = sm.Logit(y, X).fit(disp=False)

    # Pseudo R2 (McFadden)
    llf = model.llf
    llnull = model.llnull
    pseudo_r2 = 1 - llf / llnull if llnull != 0 else np.nan

    # Odds ratios per SD
    params = model.params
    conf = model.conf_int()
    or_per_sd = np.exp(params)
    or_ci = np.exp(conf)

    # Simple descriptive: win rates by advantages
    df['size_adv'] = np.where(df['rel_size'] > 0, 'focal_larger',
                      np.where(df['rel_size'] < 0, 'focal_smaller', 'equal'))
    df['loc_adv_cat'] = np.where(df['loc_adv'] > 0, 'closer_to_focal',
                          np.where(df['loc_adv'] < 0, 'closer_to_other', 'equal'))

    size_rates = df.groupby('size_adv')['win'].mean().to_dict()
    loc_rates = df.groupby('loc_adv_cat')['win'].mean().to_dict()

    # Save results to json for easy reading
    results = {
        'n': int(df.shape[0]),
        'model_summary': {
            'params': model.params.to_dict(),
            'pvalues': model.pvalues.to_dict(),
            'conf_int': conf.to_dict(),
            'odds_ratio_per_sd': or_per_sd.to_dict(),
            'odds_ratio_ci': or_ci.to_dict(),
            'pseudo_r2_mcfadden': float(pseudo_r2),
        },
        'descriptive': {
            'win_rate_by_size_adv': size_rates,
            'win_rate_by_location_adv': loc_rates,
        }
    }

    with open('analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
