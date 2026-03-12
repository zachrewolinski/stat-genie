import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
csv_path = "soccer.csv"
df = pd.read_csv(csv_path)

# Average skin tone across raters (0-1 scale with 5 discrete levels)
df = df.copy()
df['skin_mean'] = df[['rater1', 'rater2']].mean(axis=1)

# Define light and dark groups based on scale endpoints
# Light: very light or light (<= 0.25)
# Dark: dark or very dark (>= 0.75)
df['is_light'] = df['skin_mean'] <= 0.25
df['is_dark'] = df['skin_mean'] >= 0.75

# Filter to rows with skin ratings and positive games
use = df['skin_mean'].notna() & df['games'].notna() & (df['games'] > 0)
sub = df.loc[use].copy()

# Compare red-card rates per game for light vs dark groups
light = sub[sub['is_light']]
dark = sub[sub['is_dark']]

summary = {}
summary['n_rows_total'] = int(len(sub))
summary['n_rows_light'] = int(len(light))
summary['n_rows_dark'] = int(len(dark))

# Rates per 100 games
summary['light_reds_per_100'] = float(light['redCards'].sum() / light['games'].sum() * 100) if len(light) else np.nan
summary['dark_reds_per_100'] = float(dark['redCards'].sum() / dark['games'].sum() * 100) if len(dark) else np.nan

# Poisson regression with exposure (games) for dark vs light
# Keep only light and dark groups for a clean contrast
ld = sub[sub['is_light'] | sub['is_dark']].copy()
ld['dark_indicator'] = ld['is_dark'].astype(int)

# Use Poisson GLM with log link and offset for games
# Add constant term
X = sm.add_constant(ld['dark_indicator'])
y = ld['redCards']
# Offset is log(games)
offset = np.log(ld['games'])

poisson_model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
poisson_res = poisson_model.fit()

# Extract effect (rate ratio) and p-value
coef = poisson_res.params['dark_indicator']
se = poisson_res.bse['dark_indicator']
p_value = poisson_res.pvalues['dark_indicator']
rate_ratio = float(np.exp(coef))

summary['poisson_coef'] = float(coef)
summary['poisson_se'] = float(se)
summary['poisson_p'] = float(p_value)
summary['poisson_rate_ratio'] = rate_ratio

# Also compute a simple rate ratio from totals
if light['games'].sum() > 0 and dark['games'].sum() > 0:
    summary['rate_ratio_raw'] = float((dark['redCards'].sum() / dark['games'].sum()) / (light['redCards'].sum() / light['games'].sum()))
else:
    summary['rate_ratio_raw'] = np.nan

# Save summary for inspection
summary_df = pd.DataFrame([summary])
summary_df.to_csv("analysis_summary.csv", index=False)

# Print key results
print(summary_df.to_string(index=False))
print("\nPoisson model summary:\n")
print(poisson_res.summary())
