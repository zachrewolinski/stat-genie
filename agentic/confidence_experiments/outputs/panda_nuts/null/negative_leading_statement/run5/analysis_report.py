import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

# Load data
_df = pd.read_csv('panda_nuts.csv')

# Standardize categories
for col in ['sex', 'help', 'hammer']:
    _df[col] = _df[col].astype('category')

# Efficiency rate
_df['efficiency'] = _df['nuts_opened'] / _df['seconds']
_df['log_seconds'] = np.log(_df['seconds'])

n_rows = len(_df)

# Negative binomial GLM with offset for seconds (rate model)
formula = 'nuts_opened ~ age + C(sex) + C(help) + C(hammer)'
model_nb = smf.glm(formula=formula, data=_df,
                   family=sm.families.NegativeBinomial(alpha=1.0),
                   offset=_df['log_seconds'])
res_nb = model_nb.fit(cov_type='cluster', cov_kwds={'groups': _df['chimpanzee']})

# Also fit without hammer for robustness
formula_nohammer = 'nuts_opened ~ age + C(sex) + C(help)'
model_nb_nohammer = smf.glm(formula=formula_nohammer, data=_df,
                            family=sm.families.NegativeBinomial(alpha=1.0),
                            offset=_df['log_seconds'])
res_nb_nohammer = model_nb_nohammer.fit(cov_type='cluster', cov_kwds={'groups': _df['chimpanzee']})

# Extract key effects (IRR)
params = res_nb.params
conf = res_nb.conf_int()

rows = []
for term in ['age', 'C(sex)[T.m]', 'C(help)[T.y]']:
    coef = params[term]
    ci_low, ci_high = conf.loc[term]
    irr = float(np.exp(coef))
    irr_low = float(np.exp(ci_low))
    irr_high = float(np.exp(ci_high))
    p = float(res_nb.pvalues[term])
    rows.append((term, coef, p, irr, irr_low, irr_high))

# Linear model on efficiency (cluster-robust)
lm = smf.ols('efficiency ~ age + C(sex) + C(help) + C(hammer)', data=_df).fit(
    cov_type='cluster', cov_kwds={'groups': _df['chimpanzee']}
)

# Nonparametric comparisons for descriptive checks
# Efficiency by sex
f_eff = _df.loc[_df['sex'] == 'f', 'efficiency']
m_eff = _df.loc[_df['sex'] == 'm', 'efficiency']
sex_test = stats.mannwhitneyu(f_eff, m_eff, alternative='two-sided')

# Efficiency by help
help_eff = _df.loc[_df['help'] == 'y', 'efficiency']
nohelp_eff = _df.loc[_df['help'] == 'N', 'efficiency']
help_test = stats.mannwhitneyu(help_eff, nohelp_eff, alternative='two-sided')

# Age correlation with efficiency (Spearman)
age_corr = stats.spearmanr(_df['age'], _df['efficiency'])

summary = {
    'n_rows': n_rows,
    'nb_params': rows,
    'nb_pvalues': res_nb.pvalues.to_dict(),
    'nb_nohammer_pvalues': res_nb_nohammer.pvalues.to_dict(),
    'lm_params': lm.params.to_dict(),
    'lm_pvalues': lm.pvalues.to_dict(),
    'sex_mannwhitney_p': float(sex_test.pvalue),
    'help_mannwhitney_p': float(help_test.pvalue),
    'age_spearman_r': float(age_corr.correlation),
    'age_spearman_p': float(age_corr.pvalue),
    'efficiency_mean': float(_df['efficiency'].mean()),
    'efficiency_median': float(_df['efficiency'].median()),
}

# Print a compact summary
print('N:', summary['n_rows'])
print('Efficiency mean:', summary['efficiency_mean'], 'median:', summary['efficiency_median'])
print('Negative binomial (with hammer) key terms:')
for term, coef, p, irr, irr_low, irr_high in summary['nb_params']:
    print(term, 'coef', round(coef,4), 'p', round(p,4), 'IRR', round(irr,3),
          'CI', (round(irr_low,3), round(irr_high,3)))

print('Negative binomial p-values (no hammer):')
for k in ['age', 'C(sex)[T.m]', 'C(help)[T.y]']:
    print(k, round(summary['nb_nohammer_pvalues'][k],4))

print('Linear model p-values (efficiency):')
for k in ['age', 'C(sex)[T.m]', 'C(help)[T.y]']:
    print(k, round(summary['lm_pvalues'][k],4))

print('Mann-Whitney sex p:', round(summary['sex_mannwhitney_p'],4))
print('Mann-Whitney help p:', round(summary['help_mannwhitney_p'],4))
print('Spearman age-eff r, p:', round(summary['age_spearman_r'],3), round(summary['age_spearman_p'],4))
