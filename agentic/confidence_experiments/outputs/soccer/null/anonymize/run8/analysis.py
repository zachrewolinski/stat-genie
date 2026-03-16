import pandas as pd
import numpy as np
from statsmodels.stats.rates import test_poisson_2indep
import statsmodels.api as sm

# Load data
_df = pd.read_csv('soccer.csv')

# Create average skin tone from two raters
skin_cols = ['feature18', 'feature19']
_df['skin_avg'] = _df[skin_cols].mean(axis=1)

# Filter to rows with skin ratings and games > 0
sub = _df.dropna(subset=['skin_avg', 'feature9', 'feature16']).copy()
sub = sub[sub['feature9'] > 0]

# Determine skin scale values present
unique_skin = np.sort(sub['skin_avg'].dropna().unique())

# Define light and dark based on 5-point normalized scale: 0, 0.25, 0.5, 0.75, 1.0
# light: <=0.25 (very light/light), dark: >=0.75 (dark/very dark)
light = sub[sub['skin_avg'] <= 0.25]
dark = sub[sub['skin_avg'] >= 0.75]

# Aggregate counts and exposure
light_red = light['feature16'].sum()
light_games = light['feature9'].sum()
dark_red = dark['feature16'].sum()
dark_games = dark['feature9'].sum()

# Poisson rate ratio test (H0: rates equal). test_poisson_2indep(count1, exposure1, count2, exposure2)
rate_test = test_poisson_2indep(dark_red, dark_games, light_red, light_games, ratio_null=1, alternative='larger')

# Rate estimates per game
light_rate = light_red / light_games if light_games else np.nan
dark_rate = dark_red / dark_games if dark_games else np.nan
rate_ratio = dark_rate / light_rate if light_rate else np.nan

# Poisson regression with skin_avg continuous and offset log games
X = sm.add_constant(sub['skin_avg'])
y = sub['feature16']
model = sm.GLM(y, X, family=sm.families.Poisson(), offset=np.log(sub['feature9']))
res = model.fit()

# Extract coefficient for skin_avg
coef = res.params['skin_avg']
se = res.bse['skin_avg']
rr = np.exp(coef)
pval = res.pvalues['skin_avg']

summary = {
    'n_rows': len(_df),
    'n_used': len(sub),
    'unique_skin_avg': unique_skin.tolist(),
    'light_n': len(light),
    'dark_n': len(dark),
    'light_red': float(light_red),
    'dark_red': float(dark_red),
    'light_games': float(light_games),
    'dark_games': float(dark_games),
    'light_rate_per_game': float(light_rate),
    'dark_rate_per_game': float(dark_rate),
    'rate_ratio_dark_vs_light': float(rate_ratio),
    'rate_test_stat': float(rate_test.statistic),
    'rate_test_pvalue': float(rate_test.pvalue),
    'glm_coef_skin': float(coef),
    'glm_se_skin': float(se),
    'glm_pvalue_skin': float(pval),
    'glm_rate_ratio_0to1': float(rr),
}

print(summary)
