import pandas as pd
import numpy as np
import statsmodels.api as sm


df = pd.read_csv('soccer.csv')

# skin tone
skin_cols = ['rater1', 'nExp']
df['mean_skin'] = df[skin_cols].mean(axis=1)

# games exposure (column with max 47)
df['games_exposure'] = df['redCards']

analysis_df = df.dropna(subset=['mean_skin']).copy()
analysis_df = analysis_df[analysis_df['games_exposure'] > 0]

outcomes = {
    'yellowCards': 'yellowCards',
    'meanExp': 'meanExp',
    'total_red': None,
}
analysis_df['total_red'] = analysis_df['yellowCards'] + analysis_df['meanExp']

for name, col in outcomes.items():
    if col is None:
        col = 'total_red'
    print('\nOutcome:', name)
    # poisson
    X = sm.add_constant(analysis_df['mean_skin'])
    model = sm.GLM(analysis_df[col], X, family=sm.families.Poisson(), offset=np.log(analysis_df['games_exposure']))
    res = model.fit()
    coef = res.params['mean_skin']
    pval = res.pvalues['mean_skin']
    rr = np.exp(coef)
    print('coef', coef, 'pval', pval, 'rate_ratio', rr)

    # group rates
    light = analysis_df[analysis_df['mean_skin'] <= 0.25]
    dark = analysis_df[analysis_df['mean_skin'] >= 0.75]
    rate_light = light[col].sum() / light['games_exposure'].sum()
    rate_dark = dark[col].sum() / dark['games_exposure'].sum()
    rr_group = rate_dark / rate_light if rate_light > 0 else np.nan
    print('group rate ratio dark/light', rr_group)
