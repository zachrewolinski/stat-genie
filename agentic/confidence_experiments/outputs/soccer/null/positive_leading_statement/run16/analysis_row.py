import sys, os
for p in ["", os.getcwd()]:
    if p in sys.path:
        sys.path.remove(p)

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
_df = pd.read_csv('soccer.csv')

# Skin tone per row
_df['skin'] = _df[['rater1','rater2']].mean(axis=1)

# Drop missing skin or invalid games
_df = _df.dropna(subset=['skin', 'games', 'redCards'])
_df = _df[_df['games'] > 0].copy()

# Define dark vs light (dark >= 0.5)
_df['dark'] = (_df['skin'] >= 0.5).astype(int)

# Summary rates by group (using exposure)
summary = _df.groupby('dark').agg(
    n_rows=('playerShort','size'),
    n_players=('playerShort','nunique'),
    total_games=('games','sum'),
    total_red=('redCards','sum')
)
summary['red_per_game'] = summary['total_red'] / summary['total_games']

# Poisson with offset (cluster-robust SE by player)
_df['log_games'] = np.log(_df['games'])

poisson_dark = smf.glm('redCards ~ dark', data=_df,
                       family=sm.families.Poisson(),
                       offset=_df['log_games']).fit(cov_type='cluster', cov_kwds={'groups': _df['playerShort']})

poisson_skin = smf.glm('redCards ~ skin', data=_df,
                       family=sm.families.Poisson(),
                       offset=_df['log_games']).fit(cov_type='cluster', cov_kwds={'groups': _df['playerShort']})

# Logistic for any red card, with log(games) as covariate
_df['any_red'] = (_df['redCards'] > 0).astype(int)
logit_dark = smf.logit('any_red ~ dark + np.log(games)', data=_df).fit(disp=False, cov_type='cluster', cov_kwds={'groups': _df['playerShort']})
logit_skin = smf.logit('any_red ~ skin + np.log(games)', data=_df).fit(disp=False, cov_type='cluster', cov_kwds={'groups': _df['playerShort']})

# Write results
with open('analysis_results.txt','w') as f:
    f.write('Summary by dark (0=light,1=dark)\n')
    f.write(summary.to_string())
    f.write('\n\nPoisson (dark) params:\n')
    f.write(str(poisson_dark.params.to_dict()))
    f.write('\nPoisson (dark) pvalues:\n')
    f.write(str(poisson_dark.pvalues.to_dict()))
    f.write('\nPoisson (dark) CI:\n')
    f.write(str(poisson_dark.conf_int().to_dict()))
    f.write('\n\nPoisson (skin continuous) params:\n')
    f.write(str(poisson_skin.params.to_dict()))
    f.write('\nPoisson (skin) pvalues:\n')
    f.write(str(poisson_skin.pvalues.to_dict()))
    f.write('\nPoisson (skin) CI:\n')
    f.write(str(poisson_skin.conf_int().to_dict()))
    f.write('\n\nLogit (dark) params:\n')
    f.write(str(logit_dark.params.to_dict()))
    f.write('\nLogit (dark) pvalues:\n')
    f.write(str(logit_dark.pvalues.to_dict()))
    f.write('\nLogit (dark) CI:\n')
    f.write(str(logit_dark.conf_int().to_dict()))
    f.write('\n\nLogit (skin) params:\n')
    f.write(str(logit_skin.params.to_dict()))
    f.write('\nLogit (skin) pvalues:\n')
    f.write(str(logit_skin.pvalues.to_dict()))
    f.write('\nLogit (skin) CI:\n')
    f.write(str(logit_skin.conf_int().to_dict()))

print(summary)
print('\nPoisson dark coef', poisson_dark.params['dark'], 'p', poisson_dark.pvalues['dark'])
print('Poisson skin coef', poisson_skin.params['skin'], 'p', poisson_skin.pvalues['skin'])
print('Logit dark coef', logit_dark.params['dark'], 'p', logit_dark.pvalues['dark'])
print('Logit skin coef', logit_skin.params['skin'], 'p', logit_skin.pvalues['skin'])
