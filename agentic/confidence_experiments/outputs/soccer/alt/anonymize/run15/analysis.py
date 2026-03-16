import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
path = 'soccer.csv'
df = pd.read_csv(path)

# Skin tone: average of two raters
skin = df[['feature18','feature19']].mean(axis=1, skipna=True)

df = df.copy()
df['skin_tone'] = skin

# Keep rows with skin tone and games
# feature9 = number of games in dyad
# feature16 = number of red cards
mask = df['skin_tone'].notna() & df['feature9'].notna() & df['feature16'].notna()
df = df.loc[mask].copy()

# Basic stats
n = len(df)

# Rate of red cards per game
# Avoid division by zero: exclude rows with 0 games (shouldn't be any)
df = df[df['feature9'] > 0].copy()

# Define dark vs light using midpoint 0.5
# light: <0.5, dark: >=0.5

df['dark'] = (df['skin_tone'] >= 0.5).astype(int)

# Compute group rates
rates = df.groupby('dark').apply(lambda g: g['feature16'].sum() / g['feature9'].sum())

# Also compute per-dyad red card rate mean (not weighted)
mean_rates = df.groupby('dark').apply(lambda g: (g['feature16'] / g['feature9']).mean())

# Poisson regression with offset log(games)
# Outcome: red cards count per dyad
# Predictor: skin_tone (continuous)
# Add intercept
X = sm.add_constant(df['skin_tone'])
model = sm.GLM(df['feature16'], X, family=sm.families.Poisson(), offset=np.log(df['feature9']))
res = model.fit()

# Extract coefficient and p-value
coef = res.params['skin_tone']
se = res.bse['skin_tone']
pval = res.pvalues['skin_tone']

# Estimate rate ratio for 1 unit increase (0 to 1)
rr = np.exp(coef)

# Also test difference in rates between dark/light using rate ratio with Poisson test
# Rate ratio using counts and exposure
# dark group
counts = df.groupby('dark')['feature16'].sum()
exposure = df.groupby('dark')['feature9'].sum()

# Using scipy for test of two Poisson rates
# approximate z-test for log rate ratio
rate_dark = counts.loc[1] / exposure.loc[1]
rate_light = counts.loc[0] / exposure.loc[0]

# log rate ratio and SE
log_rr = np.log(rate_dark / rate_light) if rate_dark>0 and rate_light>0 else np.nan
se_log_rr = np.sqrt(1/counts.loc[1] + 1/counts.loc[0]) if counts.loc[1]>0 and counts.loc[0]>0 else np.nan
z = log_rr / se_log_rr if se_log_rr and not np.isnan(se_log_rr) else np.nan
pval_rate = 2*(1-stats.norm.cdf(abs(z))) if z==z else np.nan

# Quantiles for context
quantiles = df['skin_tone'].quantile([0.25,0.5,0.75]).to_dict()

summary = {
    'n_rows': n,
    'rate_light_weighted': float(rates.loc[0]),
    'rate_dark_weighted': float(rates.loc[1]),
    'mean_rate_light': float(mean_rates.loc[0]),
    'mean_rate_dark': float(mean_rates.loc[1]),
    'poisson_coef': float(coef),
    'poisson_se': float(se),
    'poisson_pval': float(pval),
    'poisson_rr': float(rr),
    'rate_ratio_dark_light': float(rate_dark / rate_light) if rate_light>0 else np.nan,
    'rate_ratio_pval': float(pval_rate),
    'skin_quantiles': quantiles,
    'counts_dark': int(counts.loc[1]),
    'counts_light': int(counts.loc[0]),
    'exposure_dark': int(exposure.loc[1]),
    'exposure_light': int(exposure.loc[0]),
}

print(summary)
