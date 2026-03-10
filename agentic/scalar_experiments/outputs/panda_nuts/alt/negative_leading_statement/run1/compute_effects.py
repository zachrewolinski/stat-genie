import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv('panda_nuts.csv')
for col in ['sex','help','hammer']:
    if col in df.columns:
        df[col] = df[col].astype('category')

df = df.dropna(subset=['nuts_opened','seconds','age','sex','help']).copy()
df['log_seconds'] = np.log(df['seconds'])

nb = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=df,
            family=sm.families.NegativeBinomial(alpha=1.0), offset=df['log_seconds']).fit(cov_type='HC3')

params = nb.params
conf = nb.conf_int()

irr = np.exp(params)
irr_ci = np.exp(conf)

out = pd.DataFrame({
    'coef': params,
    'pvalue': nb.pvalues,
    'irr': irr,
    'irr_ci_low': irr_ci[0],
    'irr_ci_high': irr_ci[1]
})

print(out)
