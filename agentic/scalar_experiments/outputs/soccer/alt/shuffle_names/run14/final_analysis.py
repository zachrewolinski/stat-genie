import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

path = 'soccer.csv'
df = pd.read_csv(path)

# Column mapping inferred from metadata/value ranges
player_id = 'photoID'
GAMES_COL = 'redCards'      # number of games in dyad
RED_COL = 'yellowCards'     # red cards (per metadata)
SKIN_COLS = ['rater1', 'nExp']  # two skin tone raters (0-1 scale)

# Per-player skin tone
skin = df.groupby(player_id)[SKIN_COLS].mean()
skin['skin_mean'] = skin[SKIN_COLS].mean(axis=1)

# Aggregate exposure and red cards per player
agg_games = df.groupby(player_id)[GAMES_COL].sum().rename('games')
agg_red = df.groupby(player_id)[RED_COL].sum().rename('red_cards')

player = pd.concat([skin[['skin_mean']], agg_games, agg_red], axis=1)
player = player.dropna(subset=['skin_mean'])
player = player[player['games'] > 0]

player['red_rate'] = player['red_cards'] / player['games']
player['dark'] = player['skin_mean'] > 0.5

# Group rates
group_summary = player.groupby('dark')[['red_cards', 'games']].sum()
group_summary['rate'] = group_summary['red_cards'] / group_summary['games']

# Poisson regression with exposure offset
X = sm.add_constant(player['skin_mean'])
model = sm.GLM(player['red_cards'], X, family=sm.families.Poisson(), offset=np.log(player['games']))
res = model.fit()
res_robust = model.fit(cov_type='HC0')

coef = res.params['skin_mean']
se = res.bse['skin_mean']
irr = float(np.exp(coef))
ci_low = float(np.exp(coef - 1.96 * se))
ci_high = float(np.exp(coef + 1.96 * se))

pval = float(res.pvalues['skin_mean'])
pval_robust = float(res_robust.pvalues['skin_mean'])

# Dark vs light indicator model
X2 = sm.add_constant(player['dark'].astype(int))
model2 = sm.GLM(player['red_cards'], X2, family=sm.families.Poisson(), offset=np.log(player['games']))
res2 = model2.fit()
irr_dark = float(np.exp(res2.params['dark']))
pval_dark = float(res2.pvalues['dark'])

# Rate ratio for dark (0.75) vs light (0.25) skin tone on 0-1 scale
rr_dark_light = float(np.exp(coef * 0.5))

# Build explanation
n_players = int(len(player))

total_red = int(player['red_cards'].sum())
total_games = int(player['games'].sum())

rate_light = float(group_summary.loc[False, 'rate'])
rate_dark = float(group_summary.loc[True, 'rate'])

explanation = (
    "Using player-referee dyads, I aggregated red-card counts and games to the player level ("
    f"{n_players} players with skin ratings). Skin tone was the mean of the two 0-1 rater scores; "
    f"red cards were taken from the metadata-labeled red-card column (values 0-2), and exposure was "
    f"the games column (values 1-47). Total red cards = {total_red} over {total_games} games. "
    f"Players classified as dark (mean skin tone > 0.5) had a higher red-card rate per game than light/medium players "
    f"({rate_dark:.6f} vs {rate_light:.6f}; rate ratio about {rate_dark / rate_light:.2f}), though the binary-group "
    f"difference was not statistically significant (Poisson offset model p={pval_dark:.3f}). "
    f"A Poisson regression using continuous skin tone with a log(games) offset showed a positive and statistically "
    f"significant association: IRR = {irr:.2f} per 1.0 increase in skin tone (95% CI {ci_low:.2f}-{ci_high:.2f}), "
    f"p={pval:.4f} (robust p={pval_robust:.4f}). This implies moving from light (~0.25) to dark (~0.75) skin tone "
    f"raises expected red-card rate by about {rr_dark_light:.2f}x (~{(rr_dark_light-1)*100:.0f}%). "
    "Overall, the evidence supports a small-to-moderate increase in red-card likelihood with darker skin tone, "
    "but the effect is modest and sensitive to how "
    "'dark' vs 'light' is dichotomized."
)

# Likert response: weak-to-moderate yes
response = 62

out = {"response": response, "explanation": explanation}

with open('conclusion.txt', 'w') as f:
    json.dump(out, f)
