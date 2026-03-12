import pandas as pd
import numpy as np
import statsmodels.api as sm

path = 'soccer.csv'
df = pd.read_csv(path)

skin1 = 'rater1'
skin2 = 'nExp'
games_col = 'redCards'
red_col = 'yellowCards'

skin = df[[skin1, skin2]].mean(axis=1)
mask = skin.notna() & df[games_col].notna() & (df[games_col] > 0) & df[red_col].notna()
sub = df.loc[mask].copy()
sub['skin'] = skin[mask]

# Group rates
def rate(group):
    return group[red_col].sum() / group[games_col].sum()

light = sub[sub['skin'] <= 0.25]
dark = sub[sub['skin'] >= 0.75]

light_rate = rate(light)
dark_rate = rate(dark)

# Poisson regression with offset
X = sm.add_constant(sub['skin'])
offset = np.log(sub[games_col])
model = sm.GLM(sub[red_col], X, family=sm.families.Poisson(), offset=offset)
res = model.fit()
coef = res.params['skin']
se = res.bse['skin']
pval = res.pvalues['skin']
ci_low, ci_high = res.conf_int().loc['skin']

rate_ratio = float(np.exp(coef))
rr_low, rr_high = float(np.exp(ci_low)), float(np.exp(ci_high))

# Overall rates
overall_rate = sub[red_col].sum() / sub[games_col].sum()

print({
    'n_total': len(df),
    'n_skin': len(sub),
    'overall_rate': overall_rate,
    'light_n': len(light),
    'dark_n': len(dark),
    'light_rate': light_rate,
    'dark_rate': dark_rate,
    'rate_ratio_per_1_skin_unit': rate_ratio,
    'rate_ratio_ci_low': rr_low,
    'rate_ratio_ci_high': rr_high,
    'p_value': pval,
    'coef': coef,
    'se': se,
})
