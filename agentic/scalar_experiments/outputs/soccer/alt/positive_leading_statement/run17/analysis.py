import json
import numpy as np
import pandas as pd
import statsmodels.api as sm


def main():
    df = pd.read_csv('soccer.csv')

    # Average skin tone from two raters
    df['skin_avg'] = df[['rater1', 'rater2']].mean(axis=1)

    # Keep rows with skin tone and games
    df = df[(df['skin_avg'].notna()) & (df['games'] > 0)]

    # Binary dark vs light (exclude neutral 0.5)
    df['skin_cat'] = np.where(df['skin_avg'] > 0.5, 'dark',
                       np.where(df['skin_avg'] < 0.5, 'light', 'neutral'))

    df_bin = df[df['skin_cat'].isin(['dark', 'light'])].copy()

    # Group rates
    grp = df_bin.groupby('skin_cat').agg(
        red_cards=('redCards', 'sum'),
        games=('games', 'sum'),
        dyads=('redCards', 'size')
    )
    grp['red_per_game'] = grp['red_cards'] / grp['games']

    # Poisson regression with offset
    df_bin['dark'] = (df_bin['skin_cat'] == 'dark').astype(int)
    X = sm.add_constant(df_bin['dark'])
    y = df_bin['redCards']
    offset = np.log(df_bin['games'])
    model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
    res = model.fit(cov_type='HC1')

    coef = res.params['dark']
    se = res.bse['dark']
    p = res.pvalues['dark']
    irr = float(np.exp(coef))
    ci_low = float(np.exp(coef - 1.96 * se))
    ci_high = float(np.exp(coef + 1.96 * se))

    # Also estimate continuous skin tone effect
    df_cont = df.copy()
    Xc = sm.add_constant(df_cont['skin_avg'])
    yc = df_cont['redCards']
    offset_c = np.log(df_cont['games'])
    model_c = sm.GLM(yc, Xc, family=sm.families.Poisson(), offset=offset_c)
    res_c = model_c.fit(cov_type='HC1')
    coef_c = res_c.params['skin_avg']
    se_c = res_c.bse['skin_avg']
    p_c = res_c.pvalues['skin_avg']
    irr_c = float(np.exp(coef_c))
    ci_low_c = float(np.exp(coef_c - 1.96 * se_c))
    ci_high_c = float(np.exp(coef_c + 1.96 * se_c))

    summary = {
        'group_rates': grp.reset_index().to_dict(orient='records'),
        'poisson_dark_vs_light': {
            'coef': float(coef),
            'se': float(se),
            'p': float(p),
            'irr': irr,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'n': int(df_bin.shape[0])
        },
        'poisson_skin_continuous': {
            'coef': float(coef_c),
            'se': float(se_c),
            'p': float(p_c),
            'irr': irr_c,
            'ci_low': ci_low_c,
            'ci_high': ci_high_c,
            'n': int(df_cont.shape[0])
        }
    }

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
