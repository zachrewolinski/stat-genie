import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv('panda_nuts.csv')
renamed = df.rename(columns={
    'nuts_opened':'sex_actual',
    'sex':'hammer_type',
    'help':'nuts_opened_actual',
    'chimpanzee':'seconds_actual',
    'seconds':'help_received'
})
renamed['sex_actual'] = renamed['sex_actual'].astype('category')
renamed['help_received'] = renamed['help_received'].astype('category')

# Poisson with offset log seconds
renamed['log_seconds'] = np.log(renamed['seconds_actual'])

poisson = smf.glm('nuts_opened_actual ~ age + C(sex_actual) + C(help_received)',
                 data=renamed, family=sm.families.Poisson(), offset=renamed['log_seconds']).fit()
print(poisson.summary())

# Check overdispersion: Pearson chi2 / df
pearson_chi2 = poisson.pearson_chi2
ratio = pearson_chi2 / poisson.df_resid
print('overdispersion ratio', ratio)

# Negative binomial
nb = smf.glm('nuts_opened_actual ~ age + C(sex_actual) + C(help_received)',
             data=renamed, family=sm.families.NegativeBinomial(alpha=1.0), offset=renamed['log_seconds']).fit()
print(nb.summary())

# Extract incidence rate ratios and CI
for name, model in [('poisson', poisson), ('nb', nb)]:
    params = model.params
    conf = model.conf_int()
    irr = np.exp(params)
    irr_ci = np.exp(conf)
    out = pd.DataFrame({
        'irr': irr,
        'ci_low': irr_ci[0],
        'ci_high': irr_ci[1],
        'p': model.pvalues
    })
    print('\n', name)
    print(out)

