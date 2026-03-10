import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('soccer.csv')

# Compute skin tone per row from rater1/rater2 (0-1 scale)
_df['skin'] = _df[['rater1', 'rater2']].mean(axis=1, skipna=True)

# Aggregate to player level
player = (
    _df.groupby('playerShort', as_index=False)
       .agg(
           skin=('skin', 'mean'),
           games=('games', 'sum'),
           redCards=('redCards', 'sum'),
           position=('position', 'first'),
           leagueCountry=('leagueCountry', 'first')
       )
)

# Keep players with skin rating and positive games
player = player[(player['skin'].notna()) & (player['games'] > 0)].copy()
player['red_per_game'] = player['redCards'] / player['games']

# Define light/dark groups based on 5-point scale (0,0.25,0.5,0.75,1)
# Light: <=0.25, Dark: >=0.75 (may be sparse; used for reference)
player['tone_group'] = np.where(player['skin'] <= 0.25, 'light',
                        np.where(player['skin'] >= 0.75, 'dark', 'mid'))

# Quantile-based light/dark groups for adequate sample sizes
q25 = player['skin'].quantile(0.25)
q75 = player['skin'].quantile(0.75)
player['tone_q'] = np.where(player['skin'] <= q25, 'light_q',
                    np.where(player['skin'] >= q75, 'dark_q', 'mid_q'))

# Summary stats by group
summary = player.groupby('tone_group').agg(
    n_players=('playerShort', 'count'),
    total_games=('games', 'sum'),
    total_red=('redCards', 'sum'),
)
summary['red_per_game'] = summary['total_red'] / summary['total_games']

# Poisson regression at player level: redCards ~ skin + offset(log(games))
player['log_games'] = np.log(player['games'])
model_cont = smf.glm(
    formula='redCards ~ skin',
    data=player,
    family=sm.families.Poisson(),
    offset=player['log_games']
).fit(cov_type='HC0')

# Poisson regression with controls (position, league)
model_ctrl = smf.glm(
    formula='redCards ~ skin + C(position) + C(leagueCountry)',
    data=player,
    family=sm.families.Poisson(),
    offset=player['log_games']
).fit(cov_type='HC0')

# Binary dark vs light only
dl = player[player['tone_group'].isin(['light', 'dark'])].copy()
dl['dark'] = (dl['tone_group'] == 'dark').astype(int)
model_dl = smf.glm(
    formula='redCards ~ dark',
    data=dl,
    family=sm.families.Poisson(),
    offset=np.log(dl['games'])
).fit(cov_type='HC0')

# Compute rate ratio for dark vs light from model_dl
rr_dl = float(np.exp(model_dl.params['dark']))

# Quantile-based dark vs light
dlq = player[player['tone_q'].isin(['light_q', 'dark_q'])].copy()
dlq['dark_q'] = (dlq['tone_q'] == 'dark_q').astype(int)
model_dlq = smf.glm(
    formula='redCards ~ dark_q',
    data=dlq,
    family=sm.families.Poisson(),
    offset=np.log(dlq['games'])
).fit(cov_type='HC0')
rr_dlq = float(np.exp(model_dlq.params['dark_q']))

# Also compute simple rate ratio using totals
rr_simple = np.nan
if {'dark','light'}.issubset(summary.index):
    if summary.loc['light', 'red_per_game'] > 0:
        rr_simple = (summary.loc['dark','red_per_game'] / summary.loc['light','red_per_game'])

summary_q = player.groupby('tone_q').agg(
    n_players=('playerShort', 'count'),
    total_games=('games', 'sum'),
    total_red=('redCards', 'sum'),
)
summary_q['red_per_game'] = summary_q['total_red'] / summary_q['total_games']

print('N rows:', _df.shape[0])
print('N players (with skin):', player.shape[0])
print('Skin quantiles (25%, 50%, 75%):', (q25, player['skin'].quantile(0.50), q75))
print('\nSkin value counts (rounded to 3 decimals):')
print(player['skin'].round(3).value_counts().sort_index())
print('\nGroup summary (light/mid/dark):')
print(summary)
print('\nGroup summary (light_q/mid_q/dark_q):')
print(summary_q)
print('\nPoisson (skin continuous):')
print(model_cont.summary().tables[1])
print('\nPoisson (skin + controls):')
print(model_ctrl.summary().tables[1])
print('\nPoisson (dark vs light):')
print(model_dl.summary().tables[1])
print('\nPoisson (dark_q vs light_q):')
print(model_dlq.summary().tables[1])
print('\nRate ratio dark vs light (model):', rr_dl)
print('Rate ratio dark vs light (simple):', rr_simple)
print('Rate ratio dark_q vs light_q (model):', rr_dlq)
