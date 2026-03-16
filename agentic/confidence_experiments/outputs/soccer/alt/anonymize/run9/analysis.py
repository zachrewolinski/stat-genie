import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
path = 'soccer.csv'
df = pd.read_csv(path)

# Compute mean skin tone across two raters
skin = df[['feature18', 'feature19']].mean(axis=1, skipna=True)
df = df.assign(skin_tone=skin)

# Drop rows with missing skin tone or games
# feature9: number of games
# feature16: number of red cards
clean = df.dropna(subset=['skin_tone', 'feature9', 'feature16']).copy()
clean = clean[clean['feature9'] > 0]

# Aggregate by player
agg = clean.groupby('feature1', as_index=False).agg(
    skin_tone=('skin_tone', 'mean'),
    games=('feature9', 'sum'),
    red_cards=('feature16', 'sum')
)

# Rate per game
agg['red_rate'] = agg['red_cards'] / agg['games']

# Define light vs dark groups using scale endpoints
light = agg[agg['skin_tone'] <= 0.25]
dark = agg[agg['skin_tone'] >= 0.75]

# Compute group stats
light_rate = light['red_rate'].mean()
dark_rate = dark['red_rate'].mean()

# Two-sample t-test on rates (unequal variances)
# Note: rates may be non-normal; also do Mann-Whitney U
if len(light) > 1 and len(dark) > 1:
    t_stat, t_p = stats.ttest_ind(dark['red_rate'], light['red_rate'], equal_var=False, nan_policy='omit')
    try:
        u_stat, u_p = stats.mannwhitneyu(dark['red_rate'], light['red_rate'], alternative='greater')
    except ValueError:
        u_stat, u_p = np.nan, np.nan
else:
    t_stat, t_p, u_stat, u_p = np.nan, np.nan, np.nan, np.nan

# Poisson regression at player level with offset log(games)
agg['log_games'] = np.log(agg['games'])
model = smf.glm(
    formula='red_cards ~ skin_tone',
    data=agg,
    family=sm.families.Poisson(),
    offset=agg['log_games']
).fit()

coef = model.params['skin_tone']
p_value = model.pvalues['skin_tone']

# Rate ratio for a 0.5 increase in skin tone (light -> dark on 5-point scale)
rate_ratio = float(np.exp(coef * 0.5))

# Also compute predicted rate difference between skin_tone=0.25 and 0.75
# using model (per game)
intercept = model.params['Intercept']
rate_light = np.exp(intercept + coef * 0.25)
rate_dark = np.exp(intercept + coef * 0.75)

results = {
    'n_rows': int(len(clean)),
    'n_players': int(len(agg)),
    'n_light_players': int(len(light)),
    'n_dark_players': int(len(dark)),
    'light_rate_mean': float(light_rate),
    'dark_rate_mean': float(dark_rate),
    't_p': float(t_p) if t_p == t_p else None,
    'u_p': float(u_p) if u_p == u_p else None,
    'coef_skin_tone': float(coef),
    'p_value_skin_tone': float(p_value),
    'rate_ratio_0_5': rate_ratio,
    'pred_rate_light_0_25': float(rate_light),
    'pred_rate_dark_0_75': float(rate_dark)
}

print(json.dumps(results, indent=2))
