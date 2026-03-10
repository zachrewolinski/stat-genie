import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'soccer.csv'
df = pd.read_csv(path)

# Column mapping from info.json descriptions
# Skin tone ratings: rater1, nExp
# Red cards count: yellowCards (per description)
# Games in dyad (exposure): redCards

# Clean relevant columns
cols = ['rater1', 'nExp', 'yellowCards', 'redCards']
missing = [c for c in cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

sub = df[cols].copy()

# Coerce to numeric
for c in cols:
    sub[c] = pd.to_numeric(sub[c], errors='coerce')

sub = sub.dropna()

# Build mean skin tone score
sub['skin_mean'] = sub[['rater1', 'nExp']].mean(axis=1)

# Ensure exposure > 0
sub = sub[sub['redCards'] > 0]

# Create light/dark groups by median and by threshold (>=0.5)
median_skin = sub['skin_mean'].median()
sub['skin_dark_median'] = (sub['skin_mean'] >= median_skin).astype(int)
sub['skin_dark_05'] = (sub['skin_mean'] >= 0.5).astype(int)

# Compute red card rate per game
sub['red_rate'] = sub['yellowCards'] / sub['redCards']

# Group summaries
summary_median = sub.groupby('skin_dark_median').agg(
    n=('yellowCards','size'),
    total_red=('yellowCards','sum'),
    total_games=('redCards','sum'),
    mean_rate=('red_rate','mean')
)
summary_median['rate_per_game'] = summary_median['total_red'] / summary_median['total_games']

summary_05 = sub.groupby('skin_dark_05').agg(
    n=('yellowCards','size'),
    total_red=('yellowCards','sum'),
    total_games=('redCards','sum'),
    mean_rate=('red_rate','mean')
)
summary_05['rate_per_game'] = summary_05['total_red'] / summary_05['total_games']

# Poisson regression with log(games) offset
sub['log_games'] = np.log(sub['redCards'])

# Use continuous skin_mean
model = smf.glm(
    formula='yellowCards ~ skin_mean',
    data=sub,
    family=sm.families.Poisson(),
    offset=sub['log_games']
).fit()

# Also try negative binomial as robustness (if convergence)
try:
    nb_model = smf.glm(
        formula='yellowCards ~ skin_mean',
        data=sub,
        family=sm.families.NegativeBinomial(alpha=1.0),
        offset=sub['log_games']
    ).fit()
except Exception as e:
    nb_model = None

# Print key results
print('N rows:', len(sub))
print('Skin mean median:', median_skin)
print('\nGroup summary (median split):')
print(summary_median)
print('\nGroup summary (>=0.5 split):')
print(summary_05)

print('\nPoisson GLM with offset:')
print(model.summary())

if nb_model is not None:
    print('\nNegBin GLM with offset:')
    print(nb_model.summary())
