import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

sub = pd.read_csv('reading.csv')
sub = sub[sub['dyslexia_bin'] == 1].copy()
sub = sub.dropna(subset=['speed','reader_view','uuid'])
sub['log_speed'] = np.log(sub['speed'].clip(lower=1e-6))

md = smf.mixedlm('log_speed ~ reader_view', sub, groups=sub['uuid'])
res = md.fit(reml=False, method='lbfgs')
print(res.summary())
