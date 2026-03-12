import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'soccer.csv'
df = pd.read_csv(path)

# Key columns
red_cards = df['feature16']
games = df['feature9']

# Skin tone ratings (0 to 1 scale)
skin_tone = df[['feature18', 'feature19']].mean(axis=1)

# Build analysis dataframe
analysis = pd.DataFrame({
    'red_cards': red_cards,
    'games': games,
    'skin_tone': skin_tone,
})

# Keep rows with non-missing values and positive games
analysis = analysis.dropna(subset=['red_cards', 'games', 'skin_tone'])
analysis = analysis[analysis['games'] > 0]

# Basic descriptive stats
summary = {
    'n': len(analysis),
    'skin_tone_min': float(analysis['skin_tone'].min()),
    'skin_tone_max': float(analysis['skin_tone'].max()),
    'mean_red_cards': float(analysis['red_cards'].mean()),
    'mean_games': float(analysis['games'].mean()),
}

# Define light vs dark groups based on skin_tone
# Light: <= 0.25, Dark: >= 0.75 (extremes)
light = analysis[analysis['skin_tone'] <= 0.25]
dark = analysis[analysis['skin_tone'] >= 0.75]

# Rates per game and probability of any red card in dyad

def group_stats(df_group):
    if len(df_group) == 0:
        return {
            'n': 0,
            'mean_red_cards': np.nan,
            'mean_games': np.nan,
            'rate_per_game': np.nan,
            'any_red_card_rate': np.nan,
        }
    rate_per_game = df_group['red_cards'].sum() / df_group['games'].sum()
    any_red_card_rate = (df_group['red_cards'] > 0).mean()
    return {
        'n': int(len(df_group)),
        'mean_red_cards': float(df_group['red_cards'].mean()),
        'mean_games': float(df_group['games'].mean()),
        'rate_per_game': float(rate_per_game),
        'any_red_card_rate': float(any_red_card_rate),
    }

light_stats = group_stats(light)
dark_stats = group_stats(dark)

# Poisson regression with offset for games
X = sm.add_constant(analysis['skin_tone'])
offset = np.log(analysis['games'])
poisson_model = sm.GLM(analysis['red_cards'], X, family=sm.families.Poisson(), offset=offset)
poisson_res = poisson_model.fit(cov_type='HC1')

# Negative Binomial regression (to address overdispersion)
nb_model = sm.GLM(analysis['red_cards'], X, family=sm.families.NegativeBinomial(alpha=1.0), offset=offset)
nb_res = nb_model.fit(cov_type='HC1')

# Dispersion measure for Poisson
pearson_chi2 = poisson_res.pearson_chi2
pearson_df = poisson_res.df_resid
pearson_dispersion = pearson_chi2 / pearson_df if pearson_df > 0 else np.nan

# Rate ratio for dark vs light (skin tone 1 vs 0)
poisson_rr = float(np.exp(poisson_res.params['skin_tone']))
nb_rr = float(np.exp(nb_res.params['skin_tone']))

# Collect results
results = {
    'summary': summary,
    'light_stats': light_stats,
    'dark_stats': dark_stats,
    'poisson': {
        'coef_skin_tone': float(poisson_res.params['skin_tone']),
        'se_skin_tone': float(poisson_res.bse['skin_tone']),
        'pvalue_skin_tone': float(poisson_res.pvalues['skin_tone']),
        'rate_ratio_skin_tone': poisson_rr,
    },
    'negative_binomial': {
        'coef_skin_tone': float(nb_res.params['skin_tone']),
        'se_skin_tone': float(nb_res.bse['skin_tone']),
        'pvalue_skin_tone': float(nb_res.pvalues['skin_tone']),
        'rate_ratio_skin_tone': nb_rr,
    },
    'poisson_dispersion': float(pearson_dispersion),
}

print(results)
