import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('soccer.csv')

# Compute mean skin tone across raters
skin_cols = ['feature18', 'feature19']
# Ensure numeric
for c in skin_cols + ['feature9', 'feature16']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

df['skin_mean'] = df[skin_cols].mean(axis=1)

# Filter rows with required data
analysis_df = df[['skin_mean', 'feature9', 'feature16']].dropna()
analysis_df = analysis_df[analysis_df['feature9'] > 0]
analysis_df = analysis_df.rename(columns={'feature9': 'games', 'feature16': 'red_cards'})

# Basic summary
analysis_df['red_rate'] = analysis_df['red_cards'] / analysis_df['games']

# Group definitions for light vs dark
analysis_df['skin_group'] = pd.cut(
    analysis_df['skin_mean'],
    bins=[-np.inf, 0.25, 0.75, np.inf],
    labels=['light', 'mid', 'dark']
)

summary = analysis_df.groupby('skin_group').agg(
    n=('red_cards', 'size'),
    total_red=('red_cards', 'sum'),
    total_games=('games', 'sum'),
    mean_rate=('red_rate', 'mean')
)
summary['rate_per_game'] = summary['total_red'] / summary['total_games']

# Poisson regression with offset to model red card rate as function of skin tone (continuous)
# Use robust (HC3) standard errors
analysis_df['log_games'] = np.log(analysis_df['games'])

model_cont = smf.glm(
    formula='red_cards ~ skin_mean',
    data=analysis_df,
    family=sm.families.Poisson(),
    offset=analysis_df['log_games']
).fit(cov_type='HC3')

# Poisson regression for light vs dark groups (exclude mid)
ld_df = analysis_df[analysis_df['skin_group'].isin(['light', 'dark'])].copy()
model_ld = smf.glm(
    formula='red_cards ~ C(skin_group)',
    data=ld_df,
    family=sm.families.Poisson(),
    offset=np.log(ld_df['games'])
).fit()

# Extract results
beta_cont = model_cont.params['skin_mean']
se_cont = model_cont.bse['skin_mean']
p_cont = model_cont.pvalues['skin_mean']
rr_cont = np.exp(beta_cont)
ci_cont = np.exp(model_cont.conf_int().loc['skin_mean'])

beta_ld = model_ld.params.get('C(skin_group)[T.dark]', np.nan)
se_ld = model_ld.bse.get('C(skin_group)[T.dark]', np.nan)
p_ld = model_ld.pvalues.get('C(skin_group)[T.dark]', np.nan)
rr_ld = np.exp(beta_ld) if pd.notna(beta_ld) else np.nan
ci_ld = np.exp(model_ld.conf_int().loc['C(skin_group)[T.dark]']) if pd.notna(beta_ld) else (np.nan, np.nan)

print('Rows used:', len(analysis_df))
print('Skin group summary:\n', summary)
print('\nContinuous skin tone Poisson (offset log games):')
print('beta:', beta_cont, 'SE:', se_cont, 'p:', p_cont)
print('Rate ratio (0->1):', rr_cont, 'CI:', tuple(ci_cont))

print('\nLight vs Dark Poisson (offset log games):')
print('beta:', beta_ld, 'SE:', se_ld, 'p:', p_ld)
print('Rate ratio (dark vs light):', rr_ld, 'CI:', tuple(ci_ld))
