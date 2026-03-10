import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'soccer.csv'
df = pd.read_csv(path)

# Map columns based on metadata descriptions
player_id = 'photoID'  # short player name
skin1 = 'rater1'       # skin rating rater 1 (0-1)
skin2 = 'nExp'         # skin rating rater 2 (0-1)
games = 'redCards'     # number of games in dyad
red_cards = 'yellowCards'  # number of red cards

# Drop rows with missing skin ratings or games
use = df[[player_id, skin1, skin2, games, red_cards]].dropna().copy()
use['skin_avg'] = (use[skin1] + use[skin2]) / 2

# Aggregate to player level to avoid repeated dyads
player_agg = use.groupby(player_id, as_index=False).agg(
    red_cards_total=(red_cards, 'sum'),
    games_total=(games, 'sum'),
    skin_avg=('skin_avg', 'mean')
)

# Compute rate per game
player_agg['red_rate'] = player_agg['red_cards_total'] / player_agg['games_total']

# Categorize skin tone
player_agg['skin_group'] = np.where(player_agg['skin_avg'] <= 0.25, 'light',
                                    np.where(player_agg['skin_avg'] >= 0.75, 'dark', 'mid'))

# Group comparison (dark vs light)
comp = player_agg[player_agg['skin_group'].isin(['light', 'dark'])].copy()

summary = comp.groupby('skin_group').agg(
    players=('skin_group', 'size'),
    total_red_cards=('red_cards_total', 'sum'),
    total_games=('games_total', 'sum'),
    mean_rate=('red_rate', 'mean')
)

# Rate ratio for dark vs light using total rates
rate_dark = (summary.loc['dark', 'total_red_cards'] / summary.loc['dark', 'total_games']) if 'dark' in summary.index else np.nan
rate_light = (summary.loc['light', 'total_red_cards'] / summary.loc['light', 'total_games']) if 'light' in summary.index else np.nan
rate_ratio = rate_dark / rate_light if rate_light and rate_dark == rate_dark else np.nan

# Poisson regression with offset
# Avoid zero games (shouldn't be any)
player_agg = player_agg[player_agg['games_total'] > 0].copy()

X = sm.add_constant(player_agg['skin_avg'])
model = sm.GLM(player_agg['red_cards_total'], X, family=sm.families.Poisson(), offset=np.log(player_agg['games_total']))
result = model.fit()

coef = result.params['skin_avg']
pval = result.pvalues['skin_avg']
irr = np.exp(coef)

# Overdispersion check: Pearson chi2 / df
pearson_chi2 = result.pearson_chi2
pearson_disp = pearson_chi2 / result.df_resid

# Also a simple nonparametric comparison: bootstrap difference in mean rates
rng = np.random.default_rng(0)
if len(comp) > 0:
    light_rates = comp.loc[comp['skin_group']=='light', 'red_rate'].values
    dark_rates = comp.loc[comp['skin_group']=='dark', 'red_rate'].values
    # bootstrap difference in means
    boot_diffs = []
    for _ in range(5000):
        bl = rng.choice(light_rates, size=len(light_rates), replace=True)
        bd = rng.choice(dark_rates, size=len(dark_rates), replace=True)
        boot_diffs.append(bd.mean() - bl.mean())
    boot_diffs = np.array(boot_diffs)
    diff_mean = boot_diffs.mean()
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
else:
    diff_mean = np.nan
    ci_low = np.nan
    ci_high = np.nan

print('Player-level aggregated rows:', len(player_agg))
print('Skin group summary:\n', summary)
print('Rate dark:', rate_dark, 'Rate light:', rate_light, 'Rate ratio:', rate_ratio)
print('Poisson coef (skin_avg):', coef, 'IRR:', irr, 'p-value:', pval)
print('Poisson dispersion:', pearson_disp)
print('Bootstrap diff mean (dark-light):', diff_mean, 'CI:', (ci_low, ci_high))

