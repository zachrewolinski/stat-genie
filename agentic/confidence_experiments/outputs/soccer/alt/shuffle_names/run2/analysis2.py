import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'soccer.csv'
df = pd.read_csv(path)

# Identify columns based on data characteristics
# Skin tone ratings (0-1 in 0.25 steps) appear in rater1 and nExp
skin1 = 'rater1'
skin2 = 'nExp'

# Games exposure likely in column named 'redCards' (range 1-47)
# Red cards likely in column named 'yellowCards' (0-2)
games_col = 'redCards'
red_col = 'yellowCards'

# Build skin tone average
skin = df[[skin1, skin2]].mean(axis=1)

# Filter to rows with skin tone and valid games
mask = skin.notna() & df[games_col].notna() & (df[games_col] > 0) & df[red_col].notna()
sub = df.loc[mask].copy()
sub['skin'] = skin[mask]

# Basic group comparison: dark vs light
light = sub[sub['skin'] <= 0.25]
dark = sub[sub['skin'] >= 0.75]

# Compute rates
light_rate = (light[red_col].sum() / light[games_col].sum()) if len(light) else np.nan
dark_rate = (dark[red_col].sum() / dark[games_col].sum()) if len(dark) else np.nan

# Poisson regression with exposure (games)
# red_cards ~ skin
# Use GLM with log link and offset log(games)
X = sm.add_constant(sub['skin'])
offset = np.log(sub[games_col])
model = sm.GLM(sub[red_col], X, family=sm.families.Poisson(), offset=offset)
res = model.fit()

coef = res.params['skin']
pval = res.pvalues['skin']

# Also compute rate ratio per 1 unit skin (0-1 scale)
rate_ratio = np.exp(coef)

# Summaries
out = {
    'n_rows_total': len(df),
    'n_rows_skin': len(sub),
    'light_n': len(light),
    'dark_n': len(dark),
    'light_rate': light_rate,
    'dark_rate': dark_rate,
    'poisson_coef_skin': coef,
    'poisson_rate_ratio': rate_ratio,
    'poisson_pvalue': pval,
}

print(out)
