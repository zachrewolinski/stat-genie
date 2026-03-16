import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
path = 'soccer.csv'
df = pd.read_csv(path)

# Map shuffled columns to meanings using info.json descriptions
# games: number of games in dyad
# red_cards: number of red cards in dyad
# skin ratings: rater1 and nExp

df = df.copy()

# Extract variables
skin1 = df['rater1']
skin2 = df['nExp']
skin = skin1.combine(skin2, lambda a, b: np.nanmean([a, b]))

# The column named redCards contains number of games (per metadata description)
games = df['redCards'].astype(float)
# The column named yellowCards contains number of red cards (per metadata description)
red_cards = df['yellowCards'].astype(float)

# Filter valid rows
mask = skin.notna() & games.notna() & red_cards.notna() & (games > 0)

analysis_df = pd.DataFrame({
    'skin': skin[mask],
    'games': games[mask],
    'red_cards': red_cards[mask]
})

# Per-game red card rate
analysis_df['rate'] = analysis_df['red_cards'] / analysis_df['games']

# Define light vs dark groups based on extreme ratings
# light: <=0.25, dark: >=0.75
light = analysis_df[analysis_df['skin'] <= 0.25]
dark = analysis_df[analysis_df['skin'] >= 0.75]

# Summary stats
summary = {
    'n_total': len(analysis_df),
    'n_light': len(light),
    'n_dark': len(dark),
    'rate_light_mean': light['rate'].mean(),
    'rate_dark_mean': dark['rate'].mean(),
    'rate_light_median': light['rate'].median(),
    'rate_dark_median': dark['rate'].median(),
}

# Two-sample test on rates (Welch t-test) for light vs dark
# Note: rates are skewed; t-test for rough comparison
if len(light) > 1 and len(dark) > 1:
    t_stat, t_p = stats.ttest_ind(light['rate'], dark['rate'], equal_var=False)
else:
    t_stat, t_p = np.nan, np.nan

# Poisson regression with exposure (offset log(games))
X = sm.add_constant(analysis_df['skin'])
poisson_model = sm.GLM(
    analysis_df['red_cards'],
    X,
    family=sm.families.Poisson(),
    offset=np.log(analysis_df['games'])
)
poisson_res = poisson_model.fit(cov_type='HC3')

# Negative binomial as robustness (alpha estimated via NB2 by statsmodels)
try:
    nb_model = sm.GLM(
        analysis_df['red_cards'],
        X,
        family=sm.families.NegativeBinomial(alpha=1.0),
        offset=np.log(analysis_df['games'])
    )
    nb_res = nb_model.fit(cov_type='HC3')
except Exception:
    nb_res = None

# Collect results
results = {
    'summary': summary,
    't_test': {'t_stat': t_stat, 'p_value': t_p},
    'poisson': {
        'coef_skin': poisson_res.params['skin'],
        'p_value_skin': poisson_res.pvalues['skin'],
        'rr_skin': float(np.exp(poisson_res.params['skin'])),
        'coef_const': poisson_res.params['const']
    },
}

if nb_res is not None:
    results['neg_bin'] = {
        'coef_skin': nb_res.params['skin'],
        'p_value_skin': nb_res.pvalues['skin'],
        'rr_skin': float(np.exp(nb_res.params['skin']))
    }

# Print results for review
print(results)
