import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = 'soccer.csv'
df = pd.read_csv(csv_path)

# Skin tone: average of rater1 and rater2
# If one is missing, average available; if both missing, result NaN
skin = df[['rater1','rater2']].mean(axis=1, skipna=True)
df['skin'] = skin

# Drop rows without skin or games or redCards
base = df.dropna(subset=['skin','games','redCards']).copy()

# Basic distribution
mean_red = base['redCards'].mean()
var_red = base['redCards'].var()
mean_games = base['games'].mean()

# Create red cards per game
base['red_per_game'] = base['redCards'] / base['games'].replace(0, np.nan)

# Define light/dark categories using rater scale buckets
# Original scale values: 0,0.25,0.5,0.75,1.0 (normalized).
# Light: <=0.25, Dark: >=0.75
base['skin_cat'] = np.where(base['skin'] <= 0.25, 'light', np.where(base['skin'] >= 0.75, 'dark', 'mid'))

# Summary stats by skin_cat
summary = base.groupby('skin_cat').agg(
    n=('redCards','size'),
    mean_red=('redCards','mean'),
    mean_games=('games','mean'),
    mean_red_per_game=('red_per_game','mean'),
    red_any_rate=('redCards', lambda x: (x>0).mean()),
)

# Two-group comparison (dark vs light only)
dark = base[base['skin_cat']=='dark']
light = base[base['skin_cat']=='light']

# Poisson regression with offset log(games) using continuous skin
# Use robust (HC0) SEs to reduce overdispersion impact
base['log_games'] = np.log(base['games'].replace(0, np.nan))

poisson_model = sm.GLM(
    base['redCards'],
    sm.add_constant(base['skin']),
    family=sm.families.Poisson(),
    offset=base['log_games']
).fit(cov_type='HC0')

# Negative Binomial regression (discrete) with offset
# Add constant and skin
nb_model = sm.NegativeBinomial(
    base['redCards'],
    sm.add_constant(base['skin']),
    offset=base['log_games']
).fit(disp=False)

# Logistic regression for any red card (binary) with log(games) as covariate
base['red_any'] = (base['redCards'] > 0).astype(int)
logit_model = sm.Logit(
    base['red_any'],
    sm.add_constant(base[['skin','games']])
).fit(disp=False)

# Compute effect sizes
poisson_coef = poisson_model.params['skin']
poisson_p = poisson_model.pvalues['skin']
poisson_ir = np.exp(poisson_coef)

nb_coef = nb_model.params['skin']
nb_p = nb_model.pvalues['skin']
nb_ir = np.exp(nb_coef)

logit_coef = logit_model.params['skin']
logit_p = logit_model.pvalues['skin']
logit_or = np.exp(logit_coef)

# Difference in means for dark vs light (red per game)
mean_rpg_dark = dark['red_per_game'].mean()
mean_rpg_light = light['red_per_game'].mean()

# simple rate ratio
dark_rate = dark['redCards'].sum() / dark['games'].sum()
light_rate = light['redCards'].sum() / light['games'].sum()
rate_ratio = dark_rate / light_rate if light_rate > 0 else np.nan

# Output key results
print('Rows total:', len(df))
print('Rows with skin:', len(base))
print('Mean redCards:', mean_red)
print('Var redCards:', var_red)
print('Mean games:', mean_games)
print('\nSummary by skin_cat:\n', summary)
print('\nDark vs Light totals:')
print('Dark rows:', len(dark), 'Light rows:', len(light))
print('Dark redCards per game mean:', mean_rpg_dark)
print('Light redCards per game mean:', mean_rpg_light)
print('Dark rate:', dark_rate, 'Light rate:', light_rate, 'Rate ratio:', rate_ratio)

print('\nPoisson (offset log games) skin coef:', poisson_coef, 'IRR:', poisson_ir, 'p:', poisson_p)
print('NB (offset log games) skin coef:', nb_coef, 'IRR:', nb_ir, 'p:', nb_p)
print('Logit (any red) skin coef:', logit_coef, 'OR:', logit_or, 'p:', logit_p)
