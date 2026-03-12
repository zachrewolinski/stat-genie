import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'soccer.csv'
df = pd.read_csv(path)

# Map columns based on info.json descriptions
# Skin tone ratings
skin1 = df['rater1']
skin2 = df['nExp']  # second rater in metadata
skin_mean = (skin1 + skin2) / 2

# Red cards and games (exposure)
red_cards = df['yellowCards']  # metadata: number of red cards
exposure_games = df['redCards']  # metadata: number of games in dyad

# Build analysis dataframe
analysis_df = pd.DataFrame({
    'skin_mean': skin_mean,
    'red_cards': red_cards,
    'games': exposure_games,
})
analysis_df = analysis_df.dropna()

# Avoid zero exposure
analysis_df = analysis_df[analysis_df['games'] > 0]

# Descriptive stats by skin tone category
# Define light vs dark based on mean skin tone
analysis_df['skin_cat'] = pd.cut(
    analysis_df['skin_mean'],
    bins=[-np.inf, 0.25, 0.5, 0.75, np.inf],
    labels=['light', 'mid_light', 'mid_dark', 'dark']
)

# Focus on light vs dark comparison
light_df = analysis_df[analysis_df['skin_cat'] == 'light']
dark_df = analysis_df[analysis_df['skin_cat'] == 'dark']

def rate(df):
    return df['red_cards'].sum() / df['games'].sum()

light_rate = rate(light_df) if len(light_df) else np.nan
dark_rate = rate(dark_df) if len(dark_df) else np.nan
rate_ratio = dark_rate / light_rate if light_rate and not np.isnan(light_rate) else np.nan

# Poisson regression with log(games) offset
X = sm.add_constant(analysis_df['skin_mean'])
model = sm.GLM(
    analysis_df['red_cards'],
    X,
    family=sm.families.Poisson(),
    offset=np.log(analysis_df['games'])
)
result = model.fit(cov_type='HC0')

coef = result.params['skin_mean']
se = result.bse['skin_mean']
p_value = result.pvalues['skin_mean']

irr = np.exp(coef)

# Also fit negative binomial with fixed alpha=1.0 for robustness
nb_model = sm.GLM(
    analysis_df['red_cards'],
    X,
    family=sm.families.NegativeBinomial(alpha=1.0),
    offset=np.log(analysis_df['games'])
)
nb_result = nb_model.fit(cov_type='HC0')
nb_coef = nb_result.params['skin_mean']
nb_p = nb_result.pvalues['skin_mean']
nb_irr = np.exp(nb_coef)

# Summaries
summary = {
    'n_rows': len(analysis_df),
    'light_rows': len(light_df),
    'dark_rows': len(dark_df),
    'light_rate': light_rate,
    'dark_rate': dark_rate,
    'rate_ratio_dark_vs_light': rate_ratio,
    'poisson_coef_skin_mean': coef,
    'poisson_se_skin_mean': se,
    'poisson_p_skin_mean': p_value,
    'poisson_irr_skin_mean': irr,
    'nb_coef_skin_mean': nb_coef,
    'nb_p_skin_mean': nb_p,
    'nb_irr_skin_mean': nb_irr,
}

for k, v in summary.items():
    print(f"{k}: {v}")
