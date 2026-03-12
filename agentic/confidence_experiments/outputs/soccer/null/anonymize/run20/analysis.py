import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
df = pd.read_csv('soccer.csv')

# Ensure numeric columns
for col in ['feature9','feature16','feature18','feature19']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Skin tone average (0-1 scale)
df['skin_mean'] = df[['feature18','feature19']].mean(axis=1)

# Drop rows missing key info
analysis_df = df.dropna(subset=['skin_mean','feature9','feature16']).copy()
analysis_df = analysis_df[analysis_df['feature9'] > 0]

# Group definitions for descriptive comparison
analysis_df['skin_group'] = pd.cut(
    analysis_df['skin_mean'],
    bins=[-0.01, 0.25, 0.75, 1.01],
    labels=['light','medium','dark']
)

# Descriptive stats by group
stats = (
    analysis_df
    .groupby('skin_group')
    .agg(
        n=('feature16','size'),
        total_red=('feature16','sum'),
        total_games=('feature9','sum'),
        mean_red_per_game=('feature16', lambda x: np.nan),
    )
)
# Compute mean red per game properly using totals
stats['mean_red_per_game'] = stats['total_red'] / stats['total_games']

# Poisson regression with offset for games
X = sm.add_constant(analysis_df['skin_mean'])
model = sm.GLM(
    analysis_df['feature16'],
    X,
    family=sm.families.Poisson(),
    offset=np.log(analysis_df['feature9'])
)
result = model.fit()

coef = result.params['skin_mean']
se = result.bse['skin_mean']
pval = result.pvalues['skin_mean']

# Rate ratio comparing dark (0.75) vs light (0.25)
rate_ratio_dark_vs_light = float(np.exp(coef * (0.75 - 0.25)))

# Also compute mean skin in light/dark groups to show descriptive difference
light = analysis_df[analysis_df['skin_group'] == 'light']
dark = analysis_df[analysis_df['skin_group'] == 'dark']

light_rate = light['feature16'].sum() / light['feature9'].sum()
dark_rate = dark['feature16'].sum() / dark['feature9'].sum()

# Output summary
print('rows_used', len(analysis_df))
print('skin_mean_unique', analysis_df['skin_mean'].unique())
print('group_stats')
print(stats)
print('light_rate', light_rate)
print('dark_rate', dark_rate)
print('poisson_coef', coef)
print('poisson_se', se)
print('poisson_pval', pval)
print('rate_ratio_dark_vs_light', rate_ratio_dark_vs_light)
