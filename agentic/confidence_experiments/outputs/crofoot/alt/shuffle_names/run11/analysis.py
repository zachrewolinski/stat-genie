import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('crofoot.csv')

# Map variables based on info.json descriptions (after noticing shuffled names)
# Outcome: m_focal (0/1) -> focal win
# Group sizes: f_other (focal group size), win (other group size)
# Distances: m_other (focal distance to center), n_focal (other distance to center)

# Create relative metrics
_df['rel_group_size'] = _df['f_other'] - _df['win']  # focal minus other
_df['rel_distance'] = _df['m_other'] - _df['n_focal']  # focal distance minus other

# Logistic regression: outcome ~ relative group size + relative distance
model = smf.logit('m_focal ~ rel_group_size + rel_distance', data=_df).fit(disp=False)
model_size = smf.logit('m_focal ~ rel_group_size', data=_df).fit(disp=False)
model_dist = smf.logit('m_focal ~ rel_distance', data=_df).fit(disp=False)

# Use odds ratios and p-values

def model_info(m):
    params = m.params
    conf = m.conf_int()
    pvals = m.pvalues
    or_vals = np.exp(params)
    return pd.DataFrame({
        'coef': params,
        'odds_ratio': or_vals,
        'p_value': pvals,
        'ci_low_or': np.exp(conf[0]),
        'ci_high_or': np.exp(conf[1])
    })

info_main = model_info(model)
info_size = model_info(model_size)
info_dist = model_info(model_dist)

# Save results to a CSV for inspection
_df[['m_focal','f_other','win','m_other','n_focal','rel_group_size','rel_distance']].describe().to_csv('summary_stats.csv')
info_main.to_csv('logit_main.csv')
info_size.to_csv('logit_size.csv')
info_dist.to_csv('logit_dist.csv')

print('N', len(_df))
print('Win rate', _df['m_focal'].mean())
print('Main model')
print(info_main)
print('\nSize-only model')
print(info_size)
print('\nDist-only model')
print(info_dist)
