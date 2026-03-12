import pandas as pd
import numpy as np
import statsmodels.api as sm


df = pd.read_csv('crofoot.csv')
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

size_diff = df['f_other'] - df['win']
loc_diff = df['n_focal'] - df['m_other']

X = pd.DataFrame({'size_diff': size_diff, 'loc_diff': loc_diff})
X = sm.add_constant(X)
model = sm.Logit(df['m_focal'], X, missing='drop')
res = model.fit(disp=False)
ci = res.conf_int()

or_vals = np.exp(res.params)
ci_low = np.exp(ci[0])
ci_high = np.exp(ci[1])

print('ORs and 95% CI (diff model):')
for term in ['size_diff', 'loc_diff']:
    print(term, 'OR=%.3f, 95%% CI [%.3f, %.3f], p=%.3f' % (
        or_vals[term], ci_low[term], ci_high[term], res.pvalues[term]
    ))
