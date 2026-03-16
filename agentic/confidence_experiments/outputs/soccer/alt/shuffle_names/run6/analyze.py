import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.weightstats import DescrStatsW

# Load data
path = 'soccer.csv'
df = pd.read_csv(path)

# Column mapping based on info.json descriptions
skin1_col = 'rater1'   # skin rating rater1
skin2_col = 'nExp'     # skin rating rater2
red_cards_col = 'yellowCards'  # number of red cards in dyad
# exposure (number of games in dyad)
games_col = 'redCards'

# Prepare skin tone mean
skin = df[[skin1_col, skin2_col]].astype(float)
skin_mean = skin.mean(axis=1)

df = df.copy()
df['skin_mean'] = skin_mean

# outcome and exposure
# ensure numeric
for col in [red_cards_col, games_col]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# basic filtering: require non-missing and positive games
analysis = df.dropna(subset=['skin_mean', red_cards_col, games_col]).copy()
analysis = analysis[analysis[games_col] > 0]

# rate per game
analysis['red_rate'] = analysis[red_cards_col] / analysis[games_col]

# define dark vs light
analysis['dark'] = analysis['skin_mean'] >= 0.5

# summary stats
summary = analysis.groupby('dark').agg(
    n=('skin_mean', 'size'),
    mean_skin=('skin_mean', 'mean'),
    mean_red_cards=('yellowCards', 'mean'),
    mean_games=('redCards', 'mean'),
    mean_red_rate=('red_rate', 'mean')
)

# weighted mean red rate using games as weights
wstats = analysis.groupby('dark').apply(
    lambda g: DescrStatsW(g['red_rate'], weights=g[games_col]).mean
)

# Poisson regression with log(games) offset
# add constant and skin_mean predictor
X = sm.add_constant(analysis['skin_mean'])

# use Poisson GLM with offset
model = sm.GLM(
    analysis[red_cards_col],
    X,
    family=sm.families.Poisson(),
    offset=np.log(analysis[games_col])
)
result = model.fit(cov_type='HC1')

# also fit negative binomial to check robustness
nb_model = sm.GLM(
    analysis[red_cards_col],
    X,
    family=sm.families.NegativeBinomial(alpha=1.0),
    offset=np.log(analysis[games_col])
)
nb_result = nb_model.fit(cov_type='HC1')

# compute rate ratio for dark vs light using mean red_rate
mean_red_rate_dark = summary.loc[True, 'mean_red_rate']
mean_red_rate_light = summary.loc[False, 'mean_red_rate']
rate_ratio = mean_red_rate_dark / mean_red_rate_light if mean_red_rate_light != 0 else np.nan
rate_diff = mean_red_rate_dark - mean_red_rate_light

# output key results
print('Rows total:', len(df))
print('Rows analysis:', len(analysis))
print('Summary by dark vs light:\n', summary)
print('Weighted mean red rate (games weighted):\n', wstats)

print('\nPoisson GLM (offset log games) HC1:')
print(result.summary())
print('\nNegative Binomial GLM (offset log games) HC1:')
print(nb_result.summary())

print('\nRate ratio (dark/light):', rate_ratio)
print('Rate diff (dark - light):', rate_diff)
