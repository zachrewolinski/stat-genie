import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

_df = pd.read_csv('panda_nuts.csv')
_df['sex'] = _df['sex'].astype('category')
_df['help'] = _df['help'].astype('category')
_df['hammer'] = _df['hammer'].astype('category')
_df['log_seconds'] = np.log(_df['seconds'])

# Discrete Negative Binomial (NB2) with offset
full_formula = 'nuts_opened ~ age + sex + help + hammer'
base_formula = 'nuts_opened ~ hammer'

full_model = smf.negativebinomial(full_formula, data=_df, offset=_df['log_seconds']).fit(disp=False)
base_model = smf.negativebinomial(base_formula, data=_df, offset=_df['log_seconds']).fit(disp=False)

# Likelihood ratio test for age+sex+help
lr_stat = 2 * (full_model.llf - base_model.llf)
df_diff = full_model.df_model - base_model.df_model
lr_p = stats.chi2.sf(lr_stat, df_diff)

# Extract coefficients for age/sex/help
params = full_model.params
conf = full_model.conf_int()
pvals = full_model.pvalues
coef_table = pd.DataFrame({
    'coef': params,
    'p_value': pvals,
    'ci_low': conf[0],
    'ci_high': conf[1],
})

coef_table.to_csv('negbin2_coeffs.csv')

with open('negbin2_stats.txt', 'w') as f:
    f.write(f"LL full: {full_model.llf}\n")
    f.write(f"LL base: {base_model.llf}\n")
    f.write(f"LR stat: {lr_stat}\n")
    f.write(f"DF diff: {df_diff}\n")
    f.write(f"LR p: {lr_p}\n")
    f.write(f"Alpha: {full_model.params.get('alpha', float('nan'))}\n")

print('LR test p', lr_p)
print(coef_table.loc[[c for c in coef_table.index if c.startswith('age') or c.startswith('sex') or c.startswith('help')]])
