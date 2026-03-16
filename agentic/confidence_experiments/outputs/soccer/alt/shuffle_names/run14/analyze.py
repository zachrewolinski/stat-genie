import pandas as pd
import numpy as np
import statsmodels.api as sm

path = 'soccer.csv'
df = pd.read_csv(path)

# Identify key columns based on value ranges and constraints
player_id = 'photoID'  # unique player identifier
# Column with 1-47 and sum of three outcome columns -> number of games in dyad
GAMES_COL = 'redCards'
# Skin tone ratings (0-1 scale, 5 unique values)
SKIN_COLS = ['rater1', 'nExp']

# Candidate red card columns (very rare counts)
RED_CANDIDATES = ['meanExp', 'yellowCards']

# Build per-player dataset
skin = df.groupby(player_id)[SKIN_COLS].mean()
skin['skin_mean'] = skin[SKIN_COLS].mean(axis=1)

agg_games = df.groupby(player_id)[GAMES_COL].sum().rename('games')

results = {}

for red_col in RED_CANDIDATES:
    agg_red = df.groupby(player_id)[red_col].sum().rename('red_cards')
    player = pd.concat([skin[['skin_mean']], agg_games, agg_red], axis=1)
    player = player.dropna(subset=['skin_mean'])
    player = player[player['games'] > 0]

    # Rates
    player['red_rate'] = player['red_cards'] / player['games']

    # Dark vs light grouping (dark if > 0.5 on 0-1 scale)
    player['dark'] = player['skin_mean'] > 0.5

    # Group summary
    group_summary = player.groupby('dark')[['red_cards', 'games']].sum()
    group_summary['rate'] = group_summary['red_cards'] / group_summary['games']

    # Poisson regression with offset for exposure (games)
    X = sm.add_constant(player['skin_mean'])
    model = sm.GLM(player['red_cards'], X, family=sm.families.Poisson(), offset=np.log(player['games']))
    res = model.fit()
    res_robust = model.fit(cov_type='HC0')

    irr = float(np.exp(res.params['skin_mean']))
    pval = float(res.pvalues['skin_mean'])
    pval_robust = float(res_robust.pvalues['skin_mean'])

    # Dark vs light indicator model
    X2 = sm.add_constant(player['dark'].astype(int))
    model2 = sm.GLM(player['red_cards'], X2, family=sm.families.Poisson(), offset=np.log(player['games']))
    res2 = model2.fit()
    irr_dark = float(np.exp(res2.params['dark']))
    pval_dark = float(res2.pvalues['dark'])

    results[red_col] = {
        'n_players': int(len(player)),
        'total_red_cards': float(player['red_cards'].sum()),
        'total_games': float(player['games'].sum()),
        'rate_overall': float(player['red_rate'].mean()),
        'group_summary': group_summary.to_dict(),
        'irr_per_1_skin': irr,
        'pval_skin': pval,
        'pval_skin_robust': pval_robust,
        'irr_dark_vs_light': irr_dark,
        'pval_dark': pval_dark,
    }

# Print results
for col, res in results.items():
    print('\n=== Candidate red card column:', col, '===')
    for k, v in res.items():
        if k == 'group_summary':
            print('group_summary', v)
        else:
            print(k, v)
