import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('soccer.csv')

# Skin tone: average of rater1 and rater2
rater_cols = ['rater1', 'rater2']
skin = _df[rater_cols].mean(axis=1)
# If both NaN, keep NaN
skin[_df[rater_cols].isna().all(axis=1)] = np.nan

df = _df.copy()
df['skin_mean'] = skin

# Drop rows without skin ratings
df = df.dropna(subset=['skin_mean'])

# Ensure numeric
for col in ['redCards', 'games']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Basic rates
df['red_per_game'] = df['redCards'] / df['games']
df['any_red'] = (df['redCards'] > 0).astype(int)

# Define light vs dark: median split and quartiles
median_skin = df['skin_mean'].median()
df['dark_median'] = (df['skin_mean'] > median_skin).astype(int)

q1 = df['skin_mean'].quantile(0.25)
q3 = df['skin_mean'].quantile(0.75)
df['skin_group'] = np.where(
    df['skin_mean'] <= q1,
    'light_q1',
    np.where(df['skin_mean'] >= q3, 'dark_q3', 'mid')
)

# Summary stats
summary = {
    'n_rows': int(len(df)),
    'median_skin': float(median_skin),
    'q1': float(q1),
    'q3': float(q3),
    'mean_red_per_game_light_median': float(df.loc[df['dark_median'] == 0, 'red_per_game'].mean()),
    'mean_red_per_game_dark_median': float(df.loc[df['dark_median'] == 1, 'red_per_game'].mean()),
    'any_red_rate_light_median': float(df.loc[df['dark_median'] == 0, 'any_red'].mean()),
    'any_red_rate_dark_median': float(df.loc[df['dark_median'] == 1, 'any_red'].mean()),
    'mean_red_per_game_light_q1': float(df.loc[df['skin_group'] == 'light_q1', 'red_per_game'].mean()),
    'mean_red_per_game_dark_q3': float(df.loc[df['skin_group'] == 'dark_q3', 'red_per_game'].mean()),
    'any_red_rate_light_q1': float(df.loc[df['skin_group'] == 'light_q1', 'any_red'].mean()),
    'any_red_rate_dark_q3': float(df.loc[df['skin_group'] == 'dark_q3', 'any_red'].mean()),
}

# Poisson regression with exposure games
df = df[df['games'] > 0]

poisson_model = smf.glm(
    'redCards ~ skin_mean',
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df['games'])
)
poisson_res = poisson_model.fit(cov_type='HC3')

# Logistic regression for any red card
logit_model = smf.logit('any_red ~ skin_mean', data=df)
logit_res = logit_model.fit(disp=0)

# Negative binomial (optional)
nb_model = smf.glm(
    'redCards ~ skin_mean',
    data=df,
    family=sm.families.NegativeBinomial(alpha=1.0),
    offset=np.log(df['games'])
)
nb_res = nb_model.fit(cov_type='HC3')


def coef_info(res, term='skin_mean'):
    coef = float(res.params[term])
    se = float(res.bse[term])
    p = float(res.pvalues[term])
    return coef, se, p


poisson_coef, poisson_se, poisson_p = coef_info(poisson_res)
logit_coef, logit_se, logit_p = coef_info(logit_res)
nb_coef, nb_se, nb_p = coef_info(nb_res)

# Convert to interpretable effect: Poisson IRR per 0.1 increase in skin_mean
irr_per_0_1 = float(np.exp(poisson_coef * 0.1))

# Logistic OR per 0.1 increase
or_per_0_1 = float(np.exp(logit_coef * 0.1))

# NB IRR per 0.1 increase
nb_irr_per_0_1 = float(np.exp(nb_coef * 0.1))

results = {
    'summary': summary,
    'poisson_coef': poisson_coef,
    'poisson_se': poisson_se,
    'poisson_p': poisson_p,
    'poisson_irr_per_0_1': irr_per_0_1,
    'logit_coef': logit_coef,
    'logit_se': logit_se,
    'logit_p': logit_p,
    'logit_or_per_0_1': or_per_0_1,
    'nb_coef': nb_coef,
    'nb_se': nb_se,
    'nb_p': nb_p,
    'nb_irr_per_0_1': nb_irr_per_0_1,
}

print(json.dumps(results, indent=2))
