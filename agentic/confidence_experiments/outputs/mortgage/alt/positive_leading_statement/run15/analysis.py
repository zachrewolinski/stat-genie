import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


df = pd.read_csv('mortgage.csv')

# Basic cleaning: ensure binary indicators as numeric
for col in ['female', 'accept', 'deny', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing key variables
key_cols = ['female', 'accept']
base = df.dropna(subset=key_cols).copy()

# Overall approval rates by gender
rate_table = base.groupby('female')['accept'].agg(['mean', 'count'])

# Two-proportion z-test (female=1 vs female=0)
# counts of approvals
counts = base.groupby('female')['accept'].sum()
ns = base.groupby('female')['accept'].count()
# Ensure ordering: female=0, female=1
successes = np.array([counts.get(0.0, 0.0), counts.get(1.0, 0.0)])
ns_arr = np.array([ns.get(0.0, 0.0), ns.get(1.0, 0.0)])
prop_test = sm.stats.proportions_ztest(successes, ns_arr, alternative='two-sided')

# Chi-square test of independence
contingency = pd.crosstab(base['female'], base['accept'])
chi2, chi2_p, chi2_dof, chi2_exp = stats.chi2_contingency(contingency)

# Logistic regression: accept ~ female (unadjusted)
model_unadj = smf.glm('accept ~ female', data=base, family=sm.families.Binomial())
res_unadj = model_unadj.fit(cov_type='HC3')

# Logistic regression with controls
controls = [
    'black',
    'housing_expense_ratio',
    'self_employed',
    'married',
    'mortgage_credit',
    'consumer_credit',
    'bad_history',
    'PI_ratio',
    'loan_to_value',
    'denied_PMI'
]
# Keep only columns that exist
controls = [c for c in controls if c in base.columns]

formula_adj = 'accept ~ female'
if controls:
    formula_adj += ' + ' + ' + '.join(controls)

adj_data = base.dropna(subset=['accept', 'female'] + controls).copy()
model_adj = smf.glm(formula_adj, data=adj_data, family=sm.families.Binomial())
res_adj = model_adj.fit(cov_type='HC3')

# Extract key stats
female_coef_unadj = res_unadj.params.get('female', np.nan)
female_p_unadj = res_unadj.pvalues.get('female', np.nan)

female_coef_adj = res_adj.params.get('female', np.nan)
female_p_adj = res_adj.pvalues.get('female', np.nan)

# Odds ratios and confidence intervals
or_unadj = float(np.exp(female_coef_unadj)) if np.isfinite(female_coef_unadj) else np.nan
or_adj = float(np.exp(female_coef_adj)) if np.isfinite(female_coef_adj) else np.nan

def or_ci(result, param_name='female', alpha=0.05):
    if param_name not in result.params.index:
        return (np.nan, np.nan)
    ci = result.conf_int(alpha=alpha).loc[param_name]
    return (float(np.exp(ci[0])), float(np.exp(ci[1])))

unadj_or_ci = or_ci(res_unadj, 'female')
adj_or_ci = or_ci(res_adj, 'female')

summary = {
    'n_total': int(len(base)),
    'rate_table': rate_table.reset_index().to_dict(orient='records'),
    'prop_test_z': float(prop_test[0]),
    'prop_test_p': float(prop_test[1]),
    'chi2': float(chi2),
    'chi2_p': float(chi2_p),
    'unadj_female_coef': float(female_coef_unadj),
    'unadj_female_p': float(female_p_unadj),
    'unadj_female_or': or_unadj,
    'unadj_female_or_ci': unadj_or_ci,
    'adj_formula': formula_adj,
    'adj_n': int(len(adj_data)),
    'adj_female_coef': float(female_coef_adj),
    'adj_female_p': float(female_p_adj),
    'adj_female_or': or_adj,
    'adj_female_or_ci': adj_or_ci,
}

with open('analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
