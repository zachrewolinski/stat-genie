import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

path='soccer.csv'
df=pd.read_csv(path)

# infer columns
skin1='rater1'
skin2='nExp'
red_cards='yellowCards'
# number of games between player-referee dyad
num_games='redCards'

# compute mean skin tone
mean_skin=df[[skin1, skin2]].mean(axis=1)

# dataset with skin ratings
sub=df.copy()
sub['mean_skin']=mean_skin
sub=sub.dropna(subset=['mean_skin'])

# ensure games positive
sub=sub[sub[num_games] > 0]

print('rows with skin', len(sub))

# distribution of mean_skin
print('mean_skin value counts')
print(sub['mean_skin'].value_counts().sort_index())

# red card counts
print('red card count distribution')
print(sub[red_cards].value_counts().sort_index())

# Poisson regression with offset
# small constant to avoid log(0)
sub['log_games']=np.log(sub[num_games])

model=smf.glm(f"{red_cards} ~ mean_skin", data=sub, family=sm.families.Poisson(), offset=sub['log_games']).fit()
print(model.summary())

# compute effect of skin on rate: exp(beta)
print('rate ratio for mean_skin', np.exp(model.params['mean_skin']))

# group comparison: light (<=0.25) vs dark (>=0.75)
light=sub[sub['mean_skin']<=0.25]
dark=sub[sub['mean_skin']>=0.75]

# compute red cards per game rates
light_rate=light[red_cards].sum()/light[num_games].sum()
dark_rate=dark[red_cards].sum()/dark[num_games].sum()
print('light n', len(light), 'dark n', len(dark))
print('light_rate', light_rate, 'dark_rate', dark_rate, 'rate ratio', dark_rate/light_rate if light_rate>0 else np.nan)

# rate ratio significance using poisson test
# use statsmodels for rate ratio? approximate using normal with log rate ratio
if light_rate>0 and dark_rate>0:
    # compute log rate ratio and SE
    # variance of log(rate) approx 1/rcards
    light_rc=light[red_cards].sum()
    dark_rc=dark[red_cards].sum()
    log_rr=np.log(dark_rate/light_rate)
    se=np.sqrt(1/dark_rc + 1/light_rc)
    z=log_rr/se
    p=2*(1-stats.norm.cdf(abs(z)))
    print('rate ratio z', z, 'p', p)

