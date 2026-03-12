import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import json

# Load data
_df = pd.read_csv('soccer.csv')

# Skin tone columns (rater1 and nExp are 0-1 with 5 levels)
if 'rater1' in _df.columns and 'nExp' in _df.columns:
    skin1 = _df['rater1']
    skin2 = _df['nExp']
else:
    # fallback: columns with values between 0 and 1 and ~5 unique values
    candidates = []
    for c in _df.columns:
        if pd.api.types.is_numeric_dtype(_df[c]):
            s = _df[c].dropna()
            if s.min() >= 0 and s.max() <= 1 and 3 <= s.nunique() <= 7:
                candidates.append(c)
    if len(candidates) >= 2:
        skin1 = _df[candidates[0]]
        skin2 = _df[candidates[1]]
    else:
        raise RuntimeError("Could not find skin tone columns")

# Red cards column (counts per dyad; expect max <= 2 in this dataset)
red_card_candidates = []
for c in _df.columns:
    if pd.api.types.is_numeric_dtype(_df[c]):
        s = _df[c].dropna()
        if s.min() >= 0 and s.max() <= 2 and s.nunique() <= 3:
            red_card_candidates.append(c)

# Remove skin columns from candidates
red_card_candidates = [c for c in red_card_candidates if c not in ['rater1', 'nExp']]

if not red_card_candidates:
    # fallback: allow max <= 3
    for c in _df.columns:
        if pd.api.types.is_numeric_dtype(_df[c]):
            s = _df[c].dropna()
            if s.min() >= 0 and s.max() <= 3 and s.nunique() <= 4 and c not in ['rater1','nExp']:
                red_card_candidates.append(c)

if not red_card_candidates:
    raise RuntimeError("Could not find red card column")

# Pick the candidate with smallest mean (red cards are very rare)
red_card_col = sorted(red_card_candidates, key=lambda c: _df[c].mean())[0]

# Games/exposure column (games per dyad; expected max around 47)
exposure_candidates = []
for c in _df.columns:
    if pd.api.types.is_numeric_dtype(_df[c]):
        s = _df[c].dropna()
        if s.min() >= 1 and s.max() >= 30 and s.max() <= 60:
            exposure_candidates.append(c)

if not exposure_candidates:
    # fallback: column with max between 40 and 70
    for c in _df.columns:
        if pd.api.types.is_numeric_dtype(_df[c]):
            s = _df[c].dropna()
            if s.min() >= 1 and s.max() >= 30 and s.max() <= 70:
                exposure_candidates.append(c)

if not exposure_candidates:
    raise RuntimeError("Could not find games/exposure column")

# Choose the one with max closest to 47
exposure_col = sorted(exposure_candidates, key=lambda c: abs(_df[c].max() - 47))[0]

# Build analysis frame
skin_tone = (skin1 + skin2) / 2.0
red_cards = _df[red_card_col].astype(float)
games = _df[exposure_col].astype(float)

mask = skin_tone.notna() & red_cards.notna() & games.notna() & (games > 0)
_df2 = pd.DataFrame({
    'skin_tone': skin_tone[mask],
    'red_cards': red_cards[mask],
    'games': games[mask]
})

_df2['red_card_rate'] = _df2['red_cards'] / _df2['games']

# Spearman correlation between skin tone and red card rate
spearman_r, spearman_p = stats.spearmanr(_df2['skin_tone'], _df2['red_card_rate'])

# Poisson regression with offset for games
X = sm.add_constant(_df2['skin_tone'])
model = sm.GLM(_df2['red_cards'], X, family=sm.families.Poisson(), offset=np.log(_df2['games']))
res = model.fit()
coef = res.params['skin_tone']
pval = res.pvalues['skin_tone']
irr = float(np.exp(coef))

# Quartile comparison
q1 = _df2['skin_tone'].quantile(0.25)
q3 = _df2['skin_tone'].quantile(0.75)
light = _df2[_df2['skin_tone'] <= q1]
dark = _df2[_df2['skin_tone'] >= q3]
rate_light = light['red_card_rate'].mean()
rate_dark = dark['red_card_rate'].mean()
rate_ratio = rate_dark / rate_light if rate_light > 0 else np.nan

# Welch t-test on rates
if len(light) > 1 and len(dark) > 1:
    tstat, t_p = stats.ttest_ind(dark['red_card_rate'], light['red_card_rate'], equal_var=False)
else:
    tstat, t_p = np.nan, np.nan

results = {
    'red_card_col': red_card_col,
    'exposure_col': exposure_col,
    'n': int(len(_df2)),
    'spearman_r': float(spearman_r),
    'spearman_p': float(spearman_p),
    'poisson_coef': float(coef),
    'poisson_p': float(pval),
    'irr': irr,
    'rate_light': float(rate_light),
    'rate_dark': float(rate_dark),
    'rate_ratio': float(rate_ratio),
    't_p': float(t_p) if t_p == t_p else None
}

print(json.dumps(results, indent=2))
