import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = "soccer.csv"
df = pd.read_csv(path)

# Skin tone average (use available rater values)
df['skin'] = df[['rater1','rater2']].mean(axis=1, skipna=True)

# Keep rows with skin and games>0
analysis_df = df[(~df['skin'].isna()) & (df['games'] > 0)].copy()

# Ensure redCards numeric
analysis_df['redCards'] = pd.to_numeric(analysis_df['redCards'], errors='coerce')
analysis_df = analysis_df[~analysis_df['redCards'].isna()]

# Basic counts
n = len(analysis_df)

# Poisson regression with offset(log(games))
analysis_df['log_games'] = np.log(analysis_df['games'])

X = sm.add_constant(analysis_df['skin'])
poisson_model = sm.GLM(analysis_df['redCards'], X, family=sm.families.Poisson(), offset=analysis_df['log_games'])
poisson_res = poisson_model.fit()

# Negative binomial (to check robustness)
nb_model = sm.GLM(analysis_df['redCards'], X, family=sm.families.NegativeBinomial(alpha=1.0), offset=analysis_df['log_games'])
nb_res = nb_model.fit()

# Binomial model treating redCards as successes out of games
analysis_df['red_rate'] = analysis_df['redCards'] / analysis_df['games']

binom_model = sm.GLM(analysis_df['red_rate'], X, family=sm.families.Binomial(), freq_weights=analysis_df['games'])
binom_res = binom_model.fit()

# Group comparison: light vs dark by threshold 0.5
analysis_df['dark'] = (analysis_df['skin'] >= 0.5).astype(int)

summary = analysis_df.groupby('dark').agg(
    dyads=('dark','size'),
    total_games=('games','sum'),
    total_red=('redCards','sum'),
)
summary['red_per_game'] = summary['total_red'] / summary['total_games']

# Rate ratio and difference
light_rate = summary.loc[0, 'red_per_game'] if 0 in summary.index else np.nan
dark_rate = summary.loc[1, 'red_per_game'] if 1 in summary.index else np.nan
rate_ratio = dark_rate / light_rate if (light_rate and light_rate>0) else np.nan
rate_diff = dark_rate - light_rate

# Predicted rate at skin=0 and skin=1 from Poisson model
# Using exp(b0 + b1*skin)
coef = poisson_res.params
pred_rate_light = np.exp(coef['const'] + coef['skin']*0)  # per game since offset is log(games)
pred_rate_dark = np.exp(coef['const'] + coef['skin']*1)

# Output key stats
print("N_dyads", n)
print("Poisson coef skin", poisson_res.params['skin'], "p", poisson_res.pvalues['skin'])
print("Poisson IRR", np.exp(poisson_res.params['skin']))
print("Poisson 95% CI", np.exp(poisson_res.conf_int().loc['skin']).tolist())

print("NegBin coef skin", nb_res.params['skin'], "p", nb_res.pvalues['skin'])
print("NegBin IRR", np.exp(nb_res.params['skin']))
print("NegBin 95% CI", np.exp(nb_res.conf_int().loc['skin']).tolist())

print("Binom coef skin", binom_res.params['skin'], "p", binom_res.pvalues['skin'])
print("Binom OR", np.exp(binom_res.params['skin']))
print("Binom 95% CI", np.exp(binom_res.conf_int().loc['skin']).tolist())

print("Group summary")
print(summary)
print("Rate ratio dark/light", rate_ratio)
print("Rate diff dark-light", rate_diff)
print("Pred rate skin=0", pred_rate_light)
print("Pred rate skin=1", pred_rate_dark)
