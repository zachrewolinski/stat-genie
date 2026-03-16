import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

path='soccer.csv'
df=pd.read_csv(path)

# map columns
skin1='rater1'
skin2='nExp'
red_cards='yellowCards'
num_games='redCards'

# mean skin tone
mean_skin=df[[skin1, skin2]].mean(axis=1)
df=df.copy()
df['mean_skin']=mean_skin

sub=df.dropna(subset=['mean_skin'])
sub=sub[sub[num_games] > 0]

# poisson regression with offset
sub['log_games']=np.log(sub[num_games])
model=smf.glm(f"{red_cards} ~ mean_skin", data=sub, family=sm.families.Poisson(), offset=sub['log_games']).fit()

coef=model.params['mean_skin']
pval=model.pvalues['mean_skin']
rr=np.exp(coef)
ci=np.exp(model.conf_int().loc['mean_skin'].values)

# light vs dark groups
light=sub[sub['mean_skin']<=0.25]
dark=sub[sub['mean_skin']>=0.75]

light_red=light[red_cards].sum()
dark_red=dark[red_cards].sum()
light_games=light[num_games].sum()
dark_games=dark[num_games].sum()
light_rate=light_red/light_games

dark_rate=dark_red/dark_games
rate_ratio=dark_rate/light_rate if light_rate>0 else np.nan

# poisson rate ratio significance
if light_red>0 and dark_red>0:
    log_rr=np.log(rate_ratio)
    se=np.sqrt(1/dark_red + 1/light_red)
    z=log_rr/se
    p=2*(1-stats.norm.cdf(abs(z)))
else:
    z=np.nan
    p=np.nan

print('n', len(sub))
print('mean_skin coeff', coef, 'rr', rr, 'p', pval, 'ci', ci)
print('light n', len(light), 'dark n', len(dark))
print('light red', light_red, 'dark red', dark_red)
print('light games', light_games, 'dark games', dark_games)
print('light rate', light_rate, 'dark rate', dark_rate, 'rr', rate_ratio)
print('z', z, 'p', p)
