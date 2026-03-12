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
renamed['efficiency'] = renamed['nuts_opened_actual'] / renamed['seconds_actual']

# counts
counts = {
    'n': len(renamed),
    'help_yes': int((renamed['help_received']=='y').sum()),
    'help_no': int((renamed['help_received']=='N').sum()),
    'sex_m': int((renamed['sex_actual']=='m').sum()),
    'sex_f': int((renamed['sex_actual']=='f').sum())
}

# mean efficiency by sex/help
means = renamed.groupby('sex_actual')['efficiency'].mean().to_dict()
means_help = renamed.groupby('help_received')['efficiency'].mean().to_dict()

# NB model with offset
renamed['log_seconds'] = np.log(renamed['seconds_actual'])
nb = smf.glm('nuts_opened_actual ~ age + C(sex_actual) + C(help_received)',
             data=renamed, family=sm.families.NegativeBinomial(alpha=1.0), offset=renamed['log_seconds']).fit()

params = nb.params
conf = nb.conf_int()
irr = np.exp(params)
ci = np.exp(conf)

result = {
    'counts': counts,
    'mean_eff_by_sex': means,
    'mean_eff_by_help': means_help,
    'irr_sex_m': irr['C(sex_actual)[T.m]'],
    'irr_sex_m_ci': (ci.loc['C(sex_actual)[T.m]',0], ci.loc['C(sex_actual)[T.m]',1]),
    'p_sex_m': nb.pvalues['C(sex_actual)[T.m]'],
    'irr_help_y': irr['C(help_received)[T.y]'],
    'irr_help_y_ci': (ci.loc['C(help_received)[T.y]',0], ci.loc['C(help_received)[T.y]',1]),
    'p_help_y': nb.pvalues['C(help_received)[T.y]'],
    'irr_age': irr['age'],
    'irr_age_ci': (ci.loc['age',0], ci.loc['age',1]),
    'p_age': nb.pvalues['age']
}

print(result)

