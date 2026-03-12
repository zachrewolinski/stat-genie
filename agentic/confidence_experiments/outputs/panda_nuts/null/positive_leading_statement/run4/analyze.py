import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
df = pd.read_csv('panda_nuts.csv')

# Clean / derive variables
# Efficiency: nuts opened per second (rate). Add a per-minute version for interpretability.
df['efficiency'] = df['nuts_opened'] / df['seconds']
df['eff_per_min'] = df['efficiency'] * 60

# Ensure categorical types
df['sex'] = df['sex'].astype('category')
df['help'] = df['help'].astype('category')

# Basic summary
summary = {
    'n_rows': len(df),
    'efficiency_mean': df['efficiency'].mean(),
    'efficiency_std': df['efficiency'].std(),
    'eff_per_min_mean': df['eff_per_min'].mean(),
}

# OLS model with robust (HC3) standard errors
model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit(cov_type='HC3')

# Also fit log-transformed model to reduce skew; add small constant
df['log_efficiency'] = np.log(df['efficiency'] + 1e-6)
log_model = smf.ols('log_efficiency ~ age + C(sex) + C(help)', data=df).fit(cov_type='HC3')

# Extract coefficients and p-values
def build_param_table(fit):
    return pd.DataFrame({
        'coef': fit.params,
        'std_err': fit.bse,
        'p_value': fit.pvalues,
    })

coef_table = build_param_table(model)
log_coef_table = build_param_table(log_model)

# Partial eta squared via ANOVA (type II) on base OLS (non-robust for table)
# This gives effect size, although model is the same.
try:
    anova_table = sm.stats.anova_lm(smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit(), typ=2)
except Exception:
    anova_table = None

results = {
    'summary': summary,
    'ols_params': coef_table[['coef', 'std_err', 'p_value']].to_dict(orient='index'),
    'log_ols_params': log_coef_table[['coef', 'std_err', 'p_value']].to_dict(orient='index'),
    'r2': model.rsquared,
    'adj_r2': model.rsquared_adj,
    'log_r2': log_model.rsquared,
    'log_adj_r2': log_model.rsquared_adj,
}

if anova_table is not None:
    # partial eta squared = SS_effect / (SS_effect + SS_error)
    ss_error = anova_table.loc['Residual', 'sum_sq']
    eta_sq = {}
    for term in ['age', 'C(sex)', 'C(help)']:
        if term in anova_table.index:
            ss = anova_table.loc[term, 'sum_sq']
            eta_sq[term] = float(ss / (ss + ss_error))
    results['partial_eta_sq'] = eta_sq

print(json.dumps(results, indent=2))
