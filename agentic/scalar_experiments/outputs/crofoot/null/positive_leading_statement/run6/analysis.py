import json
import numpy as np
import pandas as pd
import statsmodels.api as sm


def zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std(ddof=0)


def fit_glm(df, cols, y_col='win'):
    X = df[cols]
    X = sm.add_constant(X)
    y = df[y_col]
    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit()
    return result


def main():
    df = pd.read_csv('crofoot.csv')

    # Relative group size (focal minus other) and relative location advantage
    df['rel_size'] = df['n_focal'] - df['n_other']
    df['rel_loc'] = df['dist_other'] - df['dist_focal']  # positive => closer to focal center

    # Ratios for robustness
    df['size_ratio'] = df['n_focal'] / df['n_other']
    df['loc_ratio'] = df['dist_other'] / df['dist_focal']

    # Standardized predictors
    df['z_rel_size'] = zscore(df['rel_size'])
    df['z_rel_loc'] = zscore(df['rel_loc'])
    df['z_size_ratio'] = zscore(df['size_ratio'])
    df['z_loc_ratio'] = zscore(df['loc_ratio'])
    df['z_n_focal'] = zscore(df['n_focal'])
    df['z_n_other'] = zscore(df['n_other'])
    df['z_dist_focal'] = zscore(df['dist_focal'])
    df['z_dist_other'] = zscore(df['dist_other'])

    # Model 1: relative difference
    m1 = fit_glm(df, ['z_rel_size', 'z_rel_loc'])

    # Model 2: raw components
    m2 = fit_glm(df, ['z_n_focal', 'z_n_other', 'z_dist_focal', 'z_dist_other'])

    # Model 3: ratios
    m3 = fit_glm(df, ['z_size_ratio', 'z_loc_ratio'])

    def summarize(model, cols):
        params = model.params[cols]
        pvals = model.pvalues[cols]
        conf = model.conf_int().loc[cols]
        ors = np.exp(params)
        or_ci = np.exp(conf)
        return {
            'coef': params.to_dict(),
            'pvalues': pvals.to_dict(),
            'odds_ratio_1sd': {
                name: float(ors[name]) for name in cols
            },
            'odds_ratio_1sd_ci': {
                name: list(or_ci.loc[name].values) for name in cols
            },
            'pseudo_r2': float(1 - model.deviance / model.null_deviance),
            'aic': float(model.aic),
        }

    # Descriptive differences
    win_means = df.groupby('win')[['rel_size', 'rel_loc']].mean().rename(index={0: 'loss', 1: 'win'})

    output = {
        'n': int(df.shape[0]),
        'win_rate': float(df['win'].mean()),
        'model_rel_diff': summarize(m1, ['z_rel_size', 'z_rel_loc']),
        'model_components': summarize(m2, ['z_n_focal', 'z_n_other', 'z_dist_focal', 'z_dist_other']),
        'model_ratios': summarize(m3, ['z_size_ratio', 'z_loc_ratio']),
        'win_means': win_means.to_dict(),
        'corr': {
            'rel_size': float(df['rel_size'].corr(df['win'])),
            'rel_loc': float(df['rel_loc'].corr(df['win'])),
        },
    }

    with open('analysis_results.json', 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == '__main__':
    main()
