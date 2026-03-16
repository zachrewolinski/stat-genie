import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'soccer.csv'
df = pd.read_csv(path)

# Compute skin tone as mean of rater1 and rater2
# rater values are normalized 0-1 on a 5-point scale
skin = df[['rater1', 'rater2']].mean(axis=1)
df = df.assign(skin_tone=skin)

# Drop rows without skin tone
_df = df.dropna(subset=['skin_tone', 'redCards', 'games'])

# Aggregate to player level to reduce dependence across referees
player_agg = (
    _df.groupby('playerShort', as_index=False)
       .agg(
           skin_tone=('skin_tone', 'mean'),
           games=('games', 'sum'),
           redCards=('redCards', 'sum')
       )
)

# Basic rate comparison by grouping skin tone
# Define light (<=0.25) and dark (>=0.75)
player_agg['tone_group'] = pd.cut(
    player_agg['skin_tone'],
    bins=[-np.inf, 0.25, 0.75, np.inf],
    labels=['light', 'medium', 'dark']
)

light = player_agg[player_agg['tone_group'] == 'light']
dark = player_agg[player_agg['tone_group'] == 'dark']

# Compute red card rate per game
light_rate = light['redCards'].sum() / light['games'].sum() if light['games'].sum() > 0 else np.nan
dark_rate = dark['redCards'].sum() / dark['games'].sum() if dark['games'].sum() > 0 else np.nan

# Poisson regression at player level with offset for games
# Use skin_tone continuous
player_agg = player_agg[player_agg['games'] > 0].copy()
player_agg['log_games'] = np.log(player_agg['games'])

model = sm.GLM(
    player_agg['redCards'],
    sm.add_constant(player_agg['skin_tone']),
    family=sm.families.Poisson(),
    offset=player_agg['log_games']
)
res = model.fit(cov_type='HC0')

# Extract coefficient for skin_tone
coef = res.params['skin_tone']
se = res.bse['skin_tone']
z = coef / se if se > 0 else np.nan
p = res.pvalues['skin_tone']

# Predict rate ratio from light (0.25) to dark (0.75)
rate_ratio = np.exp(coef * (0.75 - 0.25))

# Also do a simple rate ratio test using aggregated light/dark groups
# Approximate standard error using Poisson counts
# RR = (dark_red / dark_games) / (light_red / light_games)
if light['redCards'].sum() > 0 and dark['redCards'].sum() > 0:
    rr = (dark['redCards'].sum() / dark['games'].sum()) / (light['redCards'].sum() / light['games'].sum())
    # log RR SE
    se_log_rr = np.sqrt(1/dark['redCards'].sum() + 1/light['redCards'].sum())
    z_rr = np.log(rr) / se_log_rr
else:
    rr = np.nan
    z_rr = np.nan

out = {
    'n_rows_total': len(df),
    'n_rows_with_skin': len(_df),
    'n_players_with_skin': len(player_agg),
    'light_players': len(light),
    'dark_players': len(dark),
    'light_rate': light_rate,
    'dark_rate': dark_rate,
    'glm_coef_skin': coef,
    'glm_se_skin': se,
    'glm_z_skin': z,
    'glm_p_skin': p,
    'rate_ratio_dark_vs_light_pred': rate_ratio,
    'rr_grouped': rr,
    'z_rr_grouped': z_rr,
}

print(pd.Series(out))
