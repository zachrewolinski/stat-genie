import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'soccer.csv'

df = pd.read_csv(path)

# Skin tone average
skin = df[['rater1','rater2']].mean(axis=1)
df['skin_tone'] = skin

# Keep rows with games > 0 and skin_tone not null
analysis_df = df[(df['games'] > 0) & (~df['skin_tone'].isna())].copy()

# Basic descriptive: red card rate per game by skin tone quartiles maybe
analysis_df['red_per_game'] = analysis_df['redCards'] / analysis_df['games']

# Define light/dark groups based on thresholds
light = analysis_df[analysis_df['skin_tone'] <= 0.25]
dark = analysis_df[analysis_df['skin_tone'] >= 0.75]

summary = {
    'n_rows': len(analysis_df),
    'n_players': analysis_df['playerShort'].nunique(),
    'n_light_rows': len(light),
    'n_dark_rows': len(dark),
    'red_rate_light': light['red_per_game'].mean(),
    'red_rate_dark': dark['red_per_game'].mean(),
    'red_rate_overall': analysis_df['red_per_game'].mean(),
}

# Poisson regression with offset for games
analysis_df['log_games'] = np.log(analysis_df['games'])

# Add intercept
analysis_df = analysis_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['log_games'])

# Poisson model
poisson_model = smf.glm(
    formula='redCards ~ skin_tone',
    data=analysis_df,
    family=sm.families.Poisson(),
    offset=analysis_df['log_games']
).fit()

# Negative binomial (to check overdispersion)
nb_model = smf.glm(
    formula='redCards ~ skin_tone',
    data=analysis_df,
    family=sm.families.NegativeBinomial(alpha=1.0),
    offset=analysis_df['log_games']
).fit()

# Logistic: any red card
analysis_df['any_red'] = (analysis_df['redCards'] > 0).astype(int)
logit_model = smf.glm(
    formula='any_red ~ skin_tone',
    data=analysis_df,
    family=sm.families.Binomial()
).fit()

# Compute rate ratio between skin_tone 0 and 1 for Poisson
coef = poisson_model.params['skin_tone']
se = poisson_model.bse['skin_tone']
rr = np.exp(coef)

# 95% CI for rate ratio
ci_low = np.exp(coef - 1.96*se)
ci_high = np.exp(coef + 1.96*se)

# Same for NB
coef_nb = nb_model.params['skin_tone']
se_nb = nb_model.bse['skin_tone']
rr_nb = np.exp(coef_nb)
ci_low_nb = np.exp(coef_nb - 1.96*se_nb)
ci_high_nb = np.exp(coef_nb + 1.96*se_nb)

# Logistic odds ratio
coef_logit = logit_model.params['skin_tone']
se_logit = logit_model.bse['skin_tone']
or_logit = np.exp(coef_logit)
ci_low_logit = np.exp(coef_logit - 1.96*se_logit)
ci_high_logit = np.exp(coef_logit + 1.96*se_logit)

# Output summary
print('SUMMARY')
print(summary)
print('\nPoisson coef:', coef, 'SE', se, 'p', poisson_model.pvalues['skin_tone'])
print('Rate ratio:', rr, '95% CI', (ci_low, ci_high))
print('\nNegBin coef:', coef_nb, 'SE', se_nb, 'p', nb_model.pvalues['skin_tone'])
print('Rate ratio NB:', rr_nb, '95% CI', (ci_low_nb, ci_high_nb))
print('\nLogit coef:', coef_logit, 'SE', se_logit, 'p', logit_model.pvalues['skin_tone'])
print('Odds ratio:', or_logit, '95% CI', (ci_low_logit, ci_high_logit))

# Check overdispersion for Poisson (variance/mean of residuals?)
# Compute dispersion = deviance/df_resid
print('\nPoisson dispersion (deviance/df_resid):', poisson_model.deviance / poisson_model.df_resid)

