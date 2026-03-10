import pandas as pd
import numpy as np
import statsmodels.api as sm
import json

# Load data
path = 'soccer.csv'
df = pd.read_csv(path)

# Compute average skin tone using available ratings
skin = df[['rater1', 'rater2']].mean(axis=1, skipna=True)
df = df.assign(skin=skin)

# Keep rows with skin ratings and positive games
sub = df[(df['skin'].notna()) & (df['games'] > 0)].copy()

# Extreme groups for dark vs light
light = sub[sub['skin'] <= 0.25]
dark = sub[sub['skin'] >= 0.75]

# Aggregate red cards and games for rates
summary = {}
for label, grp in [('light', light), ('dark', dark)]:
    total_red = grp['redCards'].sum()
    total_games = grp['games'].sum()
    rate = total_red / total_games if total_games > 0 else np.nan
    summary[label] = {
        'rows': len(grp),
        'total_red': float(total_red),
        'total_games': float(total_games),
        'red_per_game': float(rate) if np.isfinite(rate) else None
    }

# Poisson regression for red card rate vs skin (continuous), with offset for exposure
# Add constant and use log(games) offset
X = sm.add_constant(sub['skin'])
model = sm.GLM(sub['redCards'], X, family=sm.families.Poisson(), offset=np.log(sub['games']))
res = model.fit(cov_type='HC0')

# Extract coefficient and p-value for skin
coef_skin = res.params['skin']
se_skin = res.bse['skin']
pval_skin = res.pvalues['skin']

# Convert to rate ratio per +1 in skin (0 to 1 scale)
rate_ratio = float(np.exp(coef_skin))

# Also compute rate ratio per 0.25 (one skin-tone step)
rate_ratio_step = float(np.exp(coef_skin * 0.25))

# Store for reporting
results = {
    'n_rows': int(len(df)),
    'n_with_skin': int(len(sub)),
    'summary': summary,
    'poisson': {
        'coef_skin': float(coef_skin),
        'se_skin': float(se_skin),
        'pvalue_skin': float(pval_skin),
        'rate_ratio_per_1': rate_ratio,
        'rate_ratio_per_0_25': rate_ratio_step,
    }
}

print(json.dumps(results, indent=2))
