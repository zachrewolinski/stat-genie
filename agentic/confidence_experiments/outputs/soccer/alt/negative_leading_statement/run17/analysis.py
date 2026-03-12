import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
_df = pd.read_csv('soccer.csv')

# Basic cleaning
_df = _df[_df['games'] > 0].copy()
_df['skin_avg'] = _df[['rater1', 'rater2']].mean(axis=1)
_df = _df[~_df['skin_avg'].isna()].copy()

# Skin tone categories: light <=0.25, medium (0.25,0.75), dark >=0.75
bins = [-0.1, 0.25, 0.75, 1.1]
labels = ['light', 'medium', 'dark']
_df['skin_cat'] = pd.cut(_df['skin_avg'], bins=bins, labels=labels)

# Rate summaries
rate_summary = (
    _df.groupby('skin_cat')
    .apply(lambda g: pd.Series({
        'n_dyads': len(g),
        'total_games': g['games'].sum(),
        'total_red': g['redCards'].sum(),
        'red_per_game': g['redCards'].sum() / g['games'].sum() if g['games'].sum() > 0 else np.nan,
        'red_per_10_games': 10 * g['redCards'].sum() / g['games'].sum() if g['games'].sum() > 0 else np.nan,
    }))
)

# Poisson model: redCards ~ skin_avg with log(games) offset
X1 = sm.add_constant(_df['skin_avg'])
model1 = sm.GLM(_df['redCards'], X1, family=sm.families.Poisson(), offset=np.log(_df['games']))
res1 = model1.fit(cov_type='HC1')

# Adjusted model with covariates (league, position, height, weight, age proxy)
# Age proxy: birth year (lower means older). Convert birthday to year if possible.
_df['birth_year'] = pd.to_datetime(_df['birthday'], format='%d.%m.%Y', errors='coerce').dt.year

covariates = ['skin_avg', 'height', 'weight', 'birth_year']
cat_covs = ['leagueCountry', 'position']

X2 = _df[covariates].copy()
X2 = pd.concat([X2, pd.get_dummies(_df[cat_covs], drop_first=True)], axis=1)
X2 = sm.add_constant(X2)

# Drop rows with missing covariates
mask2 = X2.notna().all(axis=1)
model2 = sm.GLM(_df.loc[mask2, 'redCards'], X2.loc[mask2], family=sm.families.Poisson(), offset=np.log(_df.loc[mask2, 'games']))
res2 = model2.fit(cov_type='HC1')

# Binary comparison: dark vs light only
_df_dl = _df[_df['skin_cat'].isin(['light', 'dark'])].copy()
_df_dl['dark'] = (_df_dl['skin_cat'] == 'dark').astype(int)
X3 = sm.add_constant(_df_dl['dark'])
model3 = sm.GLM(_df_dl['redCards'], X3, family=sm.families.Poisson(), offset=np.log(_df_dl['games']))
res3 = model3.fit(cov_type='HC1')

# Collect key stats
coeff1 = res1.params['skin_avg']
pval1 = res1.pvalues['skin_avg']
irr1 = float(np.exp(coeff1))

coeff2 = res2.params['skin_avg']
pval2 = res2.pvalues['skin_avg']
irr2 = float(np.exp(coeff2))

coeff3 = res3.params['dark']
pval3 = res3.pvalues['dark']
irr3 = float(np.exp(coeff3))

# Build explanation
light_rate = rate_summary.loc['light', 'red_per_10_games']
medium_rate = rate_summary.loc['medium', 'red_per_10_games']
dark_rate = rate_summary.loc['dark', 'red_per_10_games']

n_total = len(_df)

explanation = (
    f"Data: {n_total} player-referee dyads with non-missing skin tone ratings and games>0. "
    f"Red-card rates per 10 games: light={light_rate:.3f}, medium={medium_rate:.3f}, dark={dark_rate:.3f}. "
    f"Poisson model with log(games) offset (unadjusted) gives IRR per 1.0 skin scale = {irr1:.3f} (p={pval1:.4g}). "
    f"Adjusted for league, position, height, weight, and birth year, IRR = {irr2:.3f} (p={pval2:.4g}). "
    f"Binary dark vs light comparison IRR = {irr3:.3f} (p={pval3:.4g}). "
    "Across specifications, statistical evidence for higher red-card rates among darker-skinned players is assessed based on p-values and effect sizes."
)

# Decide response
# Heuristic: if all pvals >= 0.05 or effects small/negative -> lean No.
if (pval1 >= 0.05) and (pval2 >= 0.05) and (pval3 >= 0.05):
    response = 20
elif (pval1 < 0.05) or (pval2 < 0.05) or (pval3 < 0.05):
    # If significant positive effect (IRR>1), lean Yes; if IRR<1, lean No.
    if (irr1 > 1.0) or (irr2 > 1.0) or (irr3 > 1.0):
        response = 70
    else:
        response = 15
else:
    response = 50

# Write conclusion
out = {"response": int(response), "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(out, f)

# Also print summary to stdout for visibility
print(json.dumps(out, indent=2))
