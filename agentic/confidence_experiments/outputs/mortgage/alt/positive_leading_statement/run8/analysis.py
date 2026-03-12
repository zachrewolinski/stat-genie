import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'mortgage.csv'
df = pd.read_csv(path)

# Basic sanity: ensure binary columns are numeric 0/1
# Outcome: deny (1=denied) or accept (1=accepted). We'll use deny as outcome.

# Compute denial rates by gender
rate_by_gender = df.groupby('female')['deny'].mean()
count_by_gender = df.groupby('female')['deny'].agg(['count','sum'])

# Chi-square test for independence (2x2)
# Build contingency table
cont_table = pd.crosstab(df['female'], df['deny'])

# Use statsmodels for chi2
from scipy.stats import chi2_contingency
chi2, p_chi2, dof, expected = chi2_contingency(cont_table)

# Unadjusted logistic regression: deny ~ female
model_unadj = smf.logit('deny ~ female', data=df).fit(disp=False)

# Adjusted logistic regression with standard covariates
# Use variables likely predictive of denial
covariates = [
    'female',
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
# Drop rows with missing in these columns
model_df = df[covariates + ['deny']].dropna()
formula = 'deny ~ ' + ' + '.join(covariates)
model_adj = smf.logit(formula, data=model_df).fit(disp=False)

# Extract odds ratios and p-values for female
unadj_coef = model_unadj.params['female']
unadj_or = float(np.exp(unadj_coef))
unadj_p = model_unadj.pvalues['female']

adj_coef = model_adj.params['female']
adj_or = float(np.exp(adj_coef))
adj_p = model_adj.pvalues['female']

# Compute average predicted denial difference (female=1 vs female=0)
# using the adjusted model and the observed covariate distribution.
model_df_f1 = model_df.copy()
model_df_f1['female'] = 1
model_df_f0 = model_df.copy()
model_df_f0['female'] = 0

pred_f1 = model_adj.predict(model_df_f1)
pred_f0 = model_adj.predict(model_df_f0)
avg_pred_diff = float((pred_f1 - pred_f0).mean())

# Also compute marginal effect summary from statsmodels for reference
try:
    margeff = model_adj.get_margeff(at='overall', method='dydx')
    meff_table = margeff.summary_frame()
    meff = float(meff_table.loc['female', 'dy/dx'])
    meff_se = float(meff_table.loc['female', 'Std. Err.'])
    meff_p = float(meff_table.loc['female', 'P>|z|'])
except Exception:
    meff = meff_se = meff_p = None

# Adjusted model for accept outcome (should be inverse sign of deny)
model_adj_accept = smf.logit('accept ~ ' + ' + '.join(covariates), data=model_df.assign(accept=1-model_df['deny'])).fit(disp=False)
adj_accept_coef = float(model_adj_accept.params['female'])
adj_accept_or = float(np.exp(adj_accept_coef))
adj_accept_p = float(model_adj_accept.pvalues['female'])

results = {
    'n': len(df),
    'rate_by_gender': rate_by_gender.to_dict(),
    'count_by_gender': count_by_gender.to_dict(),
    'chi2': chi2,
    'chi2_p': p_chi2,
    'unadj_or': unadj_or,
    'unadj_p': unadj_p,
    'adj_or': adj_or,
    'adj_p': adj_p,
    'adj_avg_pred_denial_diff_female_minus_male': avg_pred_diff,
    'adj_marginal_effect': meff,
    'adj_marginal_effect_se': meff_se,
    'adj_marginal_effect_p': meff_p,
    'adj_accept_or': adj_accept_or,
    'adj_accept_p': adj_accept_p,
    'unadj_coef': float(unadj_coef),
    'adj_coef': float(adj_coef),
}

print(json.dumps(results, indent=2))
