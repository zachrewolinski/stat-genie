import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import glm
from scipy import stats

# Load data

df = pd.read_csv('soccer.csv')

# Compute skin tone average (0-1 scale)
df['skin'] = df[['rater1', 'rater2']].mean(axis=1, skipna=True)

# Keep rows with required data
cols_needed = ['redCards', 'games', 'skin']
df = df.dropna(subset=cols_needed)

# Avoid zero games (if any)
df = df[df['games'] > 0]

# Define skin categories for light/dark comparison
# 5-point scale normalized to 0, 0.25, 0.5, 0.75, 1.0
# Use light <=0.25 and dark >=0.75 to represent clear categories

def skin_cat(x):
    if x <= 0.25:
        return 'light'
    if x >= 0.75:
        return 'dark'
    return 'mid'


df['skin_cat'] = df['skin'].apply(skin_cat)

# Binary comparison dataset
bin_df = df[df['skin_cat'].isin(['light', 'dark'])].copy()

# Aggregate rates by category
agg = bin_df.groupby('skin_cat').agg(
    red_cards=('redCards', 'sum'),
    games=('games', 'sum'),
    dyads=('redCards', 'size')
).reset_index()
agg['red_per_game'] = agg['red_cards'] / agg['games']

# Poisson regression with offset for exposure (games)
# Use log(games) as offset, predictor dark vs light
bin_df['dark'] = (bin_df['skin_cat'] == 'dark').astype(int)

poisson_model = sm.GLM(
    bin_df['redCards'],
    sm.add_constant(bin_df['dark']),
    family=sm.families.Poisson(),
    offset=np.log(bin_df['games'])
).fit(cov_type='HC3')

# Negative binomial as robustness
nb_model = sm.GLM(
    bin_df['redCards'],
    sm.add_constant(bin_df['dark']),
    family=sm.families.NegativeBinomial(alpha=1.0),
    offset=np.log(bin_df['games'])
).fit(cov_type='HC3')

# Continuous skin tone model (all data)
cont_model = sm.GLM(
    df['redCards'],
    sm.add_constant(df['skin']),
    family=sm.families.Poisson(),
    offset=np.log(df['games'])
).fit(cov_type='HC3')

# Player-level aggregation (sum over dyads)
player = df.groupby('playerShort').agg(
    red_cards=('redCards', 'sum'),
    games=('games', 'sum'),
    skin=('skin', 'mean')
).reset_index()
player = player[player['games'] > 0]
player['skin_cat'] = player['skin'].apply(skin_cat)
player_bin = player[player['skin_cat'].isin(['light', 'dark'])].copy()
player_bin['rate'] = player_bin['red_cards'] / player_bin['games']

# Nonparametric test on player-level rates
light_rates = player_bin[player_bin['skin_cat'] == 'light']['rate']
dark_rates = player_bin[player_bin['skin_cat'] == 'dark']['rate']

# Mann-Whitney U test (non-normal rates)
mwu = stats.mannwhitneyu(dark_rates, light_rates, alternative='greater')

# t-test on rates (for reference)
ttest = stats.ttest_ind(dark_rates, light_rates, equal_var=False, alternative='greater')

# Assemble summary
summary = {
    'n_total_dyads': int(len(df)),
    'n_bin_dyads': int(len(bin_df)),
    'agg_rates': agg.to_dict(orient='records'),
    'poisson_coef_dark': float(poisson_model.params['dark']),
    'poisson_se_dark': float(poisson_model.bse['dark']),
    'poisson_p_dark': float(poisson_model.pvalues['dark']),
    'poisson_irr_dark': float(np.exp(poisson_model.params['dark'])),
    'poisson_irr_ci': [
        float(np.exp(poisson_model.conf_int().loc['dark', 0])),
        float(np.exp(poisson_model.conf_int().loc['dark', 1]))
    ],
    'nb_coef_dark': float(nb_model.params['dark']),
    'nb_p_dark': float(nb_model.pvalues['dark']),
    'nb_irr_dark': float(np.exp(nb_model.params['dark'])),
    'nb_irr_ci': [
        float(np.exp(nb_model.conf_int().loc['dark', 0])),
        float(np.exp(nb_model.conf_int().loc['dark', 1]))
    ],
    'cont_coef_skin': float(cont_model.params['skin']),
    'cont_p_skin': float(cont_model.pvalues['skin']),
    'cont_irr_per_unit': float(np.exp(cont_model.params['skin'])),
    'cont_irr_ci': [
        float(np.exp(cont_model.conf_int().loc['skin', 0])),
        float(np.exp(cont_model.conf_int().loc['skin', 1]))
    ],
    'player_n_light': int((player_bin['skin_cat'] == 'light').sum()),
    'player_n_dark': int((player_bin['skin_cat'] == 'dark').sum()),
    'player_rate_light_mean': float(light_rates.mean()),
    'player_rate_dark_mean': float(dark_rates.mean()),
    'mwu_stat': float(mwu.statistic),
    'mwu_p': float(mwu.pvalue),
    'ttest_stat': float(ttest.statistic),
    'ttest_p': float(ttest.pvalue),
}

print(summary)
