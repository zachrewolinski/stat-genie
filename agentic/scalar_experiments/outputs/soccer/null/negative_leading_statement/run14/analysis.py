import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'soccer.csv'
df = pd.read_csv(path)

# Compute average skin tone
skin = df[['rater1', 'rater2']].mean(axis=1)
df = df.assign(skin_tone=skin)

# Filter valid rows
analysis_df = df.loc[(df['games'] > 0) & df['skin_tone'].notna()].copy()

# Ensure redCards non-negative
analysis_df = analysis_df[analysis_df['redCards'] >= 0]

# Rate calculations
analysis_df['red_rate'] = analysis_df['redCards'] / analysis_df['games']

# Summary statistics
n_total = len(analysis_df)
red_total = analysis_df['redCards'].sum()
games_total = analysis_df['games'].sum()

# Define light/dark groups using quartiles to reduce ambiguity
q25 = analysis_df['skin_tone'].quantile(0.25)
q75 = analysis_df['skin_tone'].quantile(0.75)
light = analysis_df[analysis_df['skin_tone'] <= q25]
dark = analysis_df[analysis_df['skin_tone'] >= q75]

# Aggregate rate per group
light_red = light['redCards'].sum()
light_games = light['games'].sum()
dark_red = dark['redCards'].sum()
dark_games = dark['games'].sum()

light_rate = light_red / light_games if light_games > 0 else np.nan
dark_rate = dark_red / dark_games if dark_games > 0 else np.nan

# Rate ratio for dark vs light
rate_ratio = (dark_rate / light_rate) if light_rate > 0 else np.nan

# Poisson regression with log(games) offset
analysis_df['log_games'] = np.log(analysis_df['games'])

# Use robust SE (HC1) for potential overdispersion
model = smf.glm(
    'redCards ~ skin_tone',
    data=analysis_df,
    family=sm.families.Poisson(),
    offset=analysis_df['log_games']
).fit(cov_type='HC1')

coef = model.params['skin_tone']
se = model.bse['skin_tone']
pval = model.pvalues['skin_tone']

# Interpret as rate ratio from 0 to 1 and from q25 to q75
rr_0_1 = np.exp(coef)
rr_q25_q75 = np.exp(coef * (q75 - q25))

# 95% CI for coef
ci_low, ci_high = model.conf_int().loc['skin_tone']
rr_0_1_ci = (np.exp(ci_low), np.exp(ci_high))

# Overdispersion check: Pearson chi2 / df
pearson_chi2 = sum(model.resid_pearson**2)
dispersion = pearson_chi2 / model.df_resid if model.df_resid > 0 else np.nan

# Save key stats for reporting
summary = {
    'n_rows': int(n_total),
    'total_red': int(red_total),
    'total_games': int(games_total),
    'q25': float(q25),
    'q75': float(q75),
    'light_rate': float(light_rate),
    'dark_rate': float(dark_rate),
    'rate_ratio_dark_vs_light': float(rate_ratio),
    'coef_skin_tone': float(coef),
    'se_skin_tone': float(se),
    'pval_skin_tone': float(pval),
    'rr_0_1': float(rr_0_1),
    'rr_0_1_ci_low': float(rr_0_1_ci[0]),
    'rr_0_1_ci_high': float(rr_0_1_ci[1]),
    'rr_q25_q75': float(rr_q25_q75),
    'dispersion': float(dispersion),
}

# Write stats for debugging
import json
with open('analysis_stats.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
