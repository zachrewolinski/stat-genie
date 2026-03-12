import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('soccer.csv')

# Compute skin tone average

df['skin_tone'] = df[['rater1', 'rater2']].mean(axis=1)

# Filter rows with skin tone and games

df = df[df['skin_tone'].notna() & df['games'].notna() & (df['games'] > 0)]

# Ensure redCards non-null

df = df[df['redCards'].notna()]

# Define groups: light <=0.25, dark >=0.75 (exclude mid)

df_group = df[(df['skin_tone'] <= 0.25) | (df['skin_tone'] >= 0.75)].copy()
df_group['dark'] = (df_group['skin_tone'] >= 0.75).astype(int)

# Aggregate rates

def rate_stats(d):
    total_games = d['games'].sum()
    total_red = d['redCards'].sum()
    rate = total_red / total_games if total_games > 0 else np.nan
    return total_games, total_red, rate

light_games, light_red, light_rate = rate_stats(df_group[df_group['dark'] == 0])
dark_games, dark_red, dark_rate = rate_stats(df_group[df_group['dark'] == 1])

# Poisson regression for rate with offset
# Use group indicator for direct comparison

model_group = smf.glm(
    formula='redCards ~ dark',
    data=df_group,
    family=sm.families.Poisson(),
    offset=np.log(df_group['games'])
).fit(cov_type='HC1')

# Continuous skin tone model (all data)

model_cont = smf.glm(
    formula='redCards ~ skin_tone',
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df['games'])
).fit(cov_type='HC1')

# Binomial model (redCards out of games) as sensitivity
# Some redCards could exceed games? Check and filter if needed

df_bin = df[df['redCards'] <= df['games']].copy()

model_bin = smf.glm(
    formula='redCards ~ skin_tone',
    data=df_bin,
    family=sm.families.Binomial(),
    freq_weights=df_bin['games']
).fit(cov_type='HC1')

# Prepare outputs

results = {
    'n_total': int(len(df)),
    'n_group': int(len(df_group)),
    'light_games': float(light_games),
    'light_red': float(light_red),
    'light_rate': float(light_rate),
    'dark_games': float(dark_games),
    'dark_red': float(dark_red),
    'dark_rate': float(dark_rate),
    'rate_ratio_dark_vs_light': float(dark_rate / light_rate) if light_rate > 0 else np.nan,
    'poisson_group_coef': float(model_group.params['dark']),
    'poisson_group_se': float(model_group.bse['dark']),
    'poisson_group_p': float(model_group.pvalues['dark']),
    'poisson_group_rr': float(np.exp(model_group.params['dark'])),
    'poisson_group_ci_low': float(np.exp(model_group.conf_int().loc['dark', 0])),
    'poisson_group_ci_high': float(np.exp(model_group.conf_int().loc['dark', 1])),
    'poisson_cont_coef': float(model_cont.params['skin_tone']),
    'poisson_cont_se': float(model_cont.bse['skin_tone']),
    'poisson_cont_p': float(model_cont.pvalues['skin_tone']),
    'poisson_cont_rr': float(np.exp(model_cont.params['skin_tone'])),
    'poisson_cont_ci_low': float(np.exp(model_cont.conf_int().loc['skin_tone', 0])),
    'poisson_cont_ci_high': float(np.exp(model_cont.conf_int().loc['skin_tone', 1])),
    'binom_cont_coef': float(model_bin.params['skin_tone']),
    'binom_cont_se': float(model_bin.bse['skin_tone']),
    'binom_cont_p': float(model_bin.pvalues['skin_tone']),
    'binom_cont_or': float(np.exp(model_bin.params['skin_tone'])),
    'binom_cont_ci_low': float(np.exp(model_bin.conf_int().loc['skin_tone', 0])),
    'binom_cont_ci_high': float(np.exp(model_bin.conf_int().loc['skin_tone', 1])),
}

for k, v in results.items():
    print(f"{k}: {v}")
