import pandas as pd
import numpy as np
import statsmodels.api as sm


df = pd.read_csv('soccer.csv')

# map fields
mean_skin = df[['rater1','nExp']].mean(axis=1)

# outcomes
red_straight = df['yellowCards']  # mapped to redCards
red_second_yellow = df['meanExp']  # mapped to yellowReds
red_total = red_straight + red_second_yellow

games = df['redCards']

# filter
mask = games > 0
_df = df[mask].copy()
_df['mean_skin'] = mean_skin[mask]
_df['red_straight'] = red_straight[mask]
_df['red_second'] = red_second_yellow[mask]
_df['red_total'] = red_total[mask]

_df = _df.dropna(subset=['mean_skin'])

bins = [-np.inf, 0.25, 0.5, np.inf]
labels = ['light','medium','dark']
_df['skin_cat'] = pd.cut(_df['mean_skin'], bins=bins, labels=labels)


def rate_table(outcome_col):
    return _df.groupby('skin_cat').apply(lambda g: pd.Series({
        'red_total': g[outcome_col].sum(),
        'games': g['redCards'].sum(),
        'rate_per_game': g[outcome_col].sum() / g['redCards'].sum()
    }))

print('Dyad-level rates (straight red):')
print(rate_table('red_straight'))
print('\nDyad-level rates (total red):')
print(rate_table('red_total'))

# poisson for straight red
X = sm.add_constant(_df['mean_skin'])
res = sm.GLM(_df['red_straight'], X, family=sm.families.Poisson(), offset=np.log(_df['redCards'])).fit()
print('\nPoisson straight red coef/p:', res.params['mean_skin'], res.pvalues['mean_skin'])

res_total = sm.GLM(_df['red_total'], X, family=sm.families.Poisson(), offset=np.log(_df['redCards'])).fit()
print('Poisson total red coef/p:', res_total.params['mean_skin'], res_total.pvalues['mean_skin'])

# rate ratio dark/light for total red
rates_total = rate_table('red_total')
rr_total = rates_total.loc['dark','rate_per_game'] / rates_total.loc['light','rate_per_game']
print('Rate ratio dark/light total:', rr_total)

rates_straight = rate_table('red_straight')
rr_straight = rates_straight.loc['dark','rate_per_game'] / rates_straight.loc['light','rate_per_game']
print('Rate ratio dark/light straight:', rr_straight)

