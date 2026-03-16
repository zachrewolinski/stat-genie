import pandas as pd
import numpy as np
import statsmodels.api as sm

path = 'soccer.csv'
df = pd.read_csv(path)

# Columns per metadata
skin1 = 'rater1'
skin2 = 'nExp'
games = 'redCards'       # number of games in dyad
red_cards = 'yellowCards'  # number of red cards

use = df[[skin1, skin2, games, red_cards]].dropna().copy()
use['skin_avg'] = (use[skin1] + use[skin2]) / 2

# Define skin groups
use['skin_group'] = np.where(use['skin_avg'] <= 0.25, 'light',
                             np.where(use['skin_avg'] >= 0.75, 'dark', 'mid'))

# Group comparison
summary = use.groupby('skin_group').agg(
    dyads=('skin_group', 'size'),
    total_red_cards=(red_cards, 'sum'),
    total_games=(games, 'sum'),
    mean_rate=(red_cards, lambda x: (x / use.loc[x.index, games]).mean())
)

rate_dark = summary.loc['dark', 'total_red_cards'] / summary.loc['dark', 'total_games']
rate_light = summary.loc['light', 'total_red_cards'] / summary.loc['light', 'total_games']
rate_ratio = rate_dark / rate_light

# Poisson regression with offset
X = sm.add_constant(use['skin_avg'])
model = sm.GLM(use[red_cards], X, family=sm.families.Poisson(), offset=np.log(use[games]))
result = model.fit()
coef = result.params['skin_avg']
irr = np.exp(coef)
pval = result.pvalues['skin_avg']

# Overdispersion
pearson_disp = result.pearson_chi2 / result.df_resid

# Nonparametric bootstrap for difference in mean rates (dark - light)
light_rates = use.loc[use['skin_group']=='light', red_cards] / use.loc[use['skin_group']=='light', games]
dark_rates = use.loc[use['skin_group']=='dark', red_cards] / use.loc[use['skin_group']=='dark', games]

rng = np.random.default_rng(0)
boot_diffs = []
for _ in range(5000):
    bl = rng.choice(light_rates.values, size=len(light_rates), replace=True)
    bd = rng.choice(dark_rates.values, size=len(dark_rates), replace=True)
    boot_diffs.append(bd.mean() - bl.mean())
boot_diffs = np.array(boot_diffs)
ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

print('Dyads:', len(use))
print('Skin group summary:\n', summary)
print('Rate dark:', rate_dark, 'Rate light:', rate_light, 'Rate ratio:', rate_ratio)
print('Poisson coef:', coef, 'IRR:', irr, 'p-value:', pval, 'dispersion:', pearson_disp)
print('Bootstrap diff (dark-light) mean:', boot_diffs.mean(), 'CI:', (ci_low, ci_high))

