import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
_df = pd.read_csv('soccer.csv')

# Column mapping based on metadata descriptions
player_id = 'photoID'
skin1 = 'rater1'
skin2 = 'nExp'
red_cards = 'yellowCards'  # described as red cards in metadata
exposure_games = 'redCards'  # described as number of games in metadata

# Compute mean skin tone per row (0-1 scale)
_df['skin_mean'] = _df[[skin1, skin2]].mean(axis=1)

# Aggregate to player level
player = (
    _df.groupby(player_id)
    .agg(
        games_total=(exposure_games, 'sum'),
        red_cards_total=(red_cards, 'sum'),
        skin_mean=('skin_mean', 'mean'),
    )
    .reset_index()
)

# Keep players with skin tone info and positive exposure
player = player[(player['skin_mean'].notna()) & (player['games_total'] > 0)]

# Define dark vs light using scale endpoints
player['skin_group'] = np.where(player['skin_mean'] >= 0.75, 'dark',
                                np.where(player['skin_mean'] <= 0.25, 'light', 'mid'))

# Group stats for light vs dark
ld = player[player['skin_group'].isin(['light', 'dark'])].copy()

group_stats = (
    ld.groupby('skin_group')
    .agg(
        players=('skin_group', 'size'),
        games_total=('games_total', 'sum'),
        red_cards_total=('red_cards_total', 'sum')
    )
)

group_stats['rate_per_game'] = group_stats['red_cards_total'] / group_stats['games_total']

# Poisson regression at player level with offset for games
ld['dark_indicator'] = (ld['skin_group'] == 'dark').astype(int)
X = sm.add_constant(ld['dark_indicator'])
y = ld['red_cards_total']
offset = np.log(ld['games_total'])

poisson_model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
poisson_res = poisson_model.fit(cov_type='HC0')

coef = poisson_res.params['dark_indicator']
se = poisson_res.bse['dark_indicator']
wald_p = poisson_res.pvalues['dark_indicator']

irr = float(np.exp(coef))

# Also compute simple rate ratio from aggregated groups
if 'dark' in group_stats.index and 'light' in group_stats.index:
    rate_dark = float(group_stats.loc['dark', 'rate_per_game'])
    rate_light = float(group_stats.loc['light', 'rate_per_game'])
    rate_ratio = rate_dark / rate_light if rate_light > 0 else np.nan
else:
    rate_dark = rate_light = rate_ratio = np.nan

# Decide response on 0-100 scale
# Criteria: if IRR>1 and p<0.05 -> evidence for higher red cards for dark skin
# scale based on effect size and p-value
if np.isnan(irr) or np.isnan(wald_p):
    response = 50
    conclusion = 'inconclusive'
elif (irr > 1) and (wald_p < 0.05):
    # stronger effect and smaller p -> higher score
    if irr >= 1.5:
        response = 80
    elif irr >= 1.2:
        response = 70
    else:
        response = 60
    conclusion = 'yes'
elif (irr < 1) and (wald_p < 0.05):
    # evidence in opposite direction
    if irr <= 0.8:
        response = 20
    else:
        response = 35
    conclusion = 'no_opposite'
else:
    # not significant
    response = 40 if irr > 1 else 30
    conclusion = 'no'

# Build explanation
explanation = (
    f"Analyzed player-referee dyads and aggregated to player level using exposure-adjusted counts. "
    f"Skin tone was computed as the mean of the two rater scores (rater1 and nExp; 0-1 scale) and players were categorized as "
    f"light (<=0.25) or dark (>=0.75). Red cards were taken from the yellowCards column (per metadata) and exposure from the redCards column (games). "
    f"Among players with light vs dark skin, red-card rates per game were {rate_light:.4f} vs {rate_dark:.4f} (rate ratio {rate_ratio:.2f}). "
    f"A Poisson regression with log(games) offset comparing dark to light players gave an incidence rate ratio of {irr:.2f} "
    f"(robust Wald p={wald_p:.4g}). "
)

if conclusion == 'yes':
    explanation += (
        "This provides statistically significant evidence that dark-skinned players receive red cards at a higher rate than light-skinned players, "
        "after accounting for exposure (games)."
    )
elif conclusion == 'no_opposite':
    explanation += (
        "This provides statistically significant evidence in the opposite direction (dark-skinned players receive red cards at a lower rate)."
    )
elif conclusion == 'no':
    explanation += (
        "The effect is not statistically significant, so the data do not provide clear evidence that dark-skinned players are more likely to receive red cards."
    )
else:
    explanation += (
        "The analysis was inconclusive due to missing or unstable estimates."
    )

# Write conclusion
with open('conclusion.txt', 'w') as f:
    json.dump({'response': int(response), 'explanation': explanation}, f)
