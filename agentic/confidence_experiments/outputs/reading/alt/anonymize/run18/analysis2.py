import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

df = pd.read_csv('reading.csv')

# reading speed
speed = df['feature20']

# dyslexia group via feature12 >= 1 (dyslexia or severe)
sub = df[df['feature12'] >= 1].copy()

on_speed = sub.loc[sub['feature3'] == 1, 'feature20']
off_speed = sub.loc[sub['feature3'] == 0, 'feature20']

print('dyslexia severity >=1 counts', len(on_speed), len(off_speed))
print('mean on', on_speed.mean(), 'mean off', off_speed.mean())
print('t-test', stats.ttest_ind(on_speed, off_speed, equal_var=False, nan_policy='omit'))

# regression controlling for covariates
sub2 = sub.dropna(subset=['feature20','feature3','feature7','feature19','feature10','feature11','feature15'])
model = smf.ols('feature20 ~ C(feature3) + feature7 + feature19 + feature10 + C(feature11) + C(feature15)', data=sub2).fit()
print(model.summary())

