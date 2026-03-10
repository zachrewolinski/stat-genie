import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data

df = pd.read_csv('mortgage.csv')

# Ensure binary columns are numeric (allow NaN)
binary_cols = ['female','black','self_employed','married','bad_history','deny','denied_PMI','accept']
for c in binary_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Basic counts
analysis_df = df.dropna(subset=['female','accept'])
n = len(analysis_df)
accept_rate = analysis_df['accept'].mean()
accept_f = analysis_df.loc[analysis_df['female']==1,'accept'].mean()
accept_m = analysis_df.loc[analysis_df['female']==0,'accept'].mean()

# Chi-square test for independence between female and accept
cont = pd.crosstab(analysis_df['female'], analysis_df['accept'])
chi2, p_chi, dof, exp = stats.chi2_contingency(cont)

# Difference in proportions (female - male)
# Use standard error for difference in proportions
p1 = accept_f
p0 = accept_m
n1 = (analysis_df['female']==1).sum()
n0 = (analysis_df['female']==0).sum()
se = np.sqrt(p1*(1-p1)/n1 + p0*(1-p0)/n0)
if se > 0:
    z = (p1 - p0) / se
    p_diff = 2*(1 - stats.norm.cdf(abs(z)))
else:
    z = np.nan
    p_diff = np.nan

# Logistic regression: accept ~ female
logit_simple = smf.logit('accept ~ female', data=analysis_df).fit(disp=False)

# Logistic regression with controls (common underwriting factors)
controls = ['black','housing_expense_ratio','self_employed','married','mortgage_credit',
            'consumer_credit','bad_history','PI_ratio','loan_to_value','denied_PMI']
# drop rows with missing
model_df = df.dropna(subset=['accept','female'] + controls)
formula = 'accept ~ female + ' + ' + '.join(controls)
logit_ctrl = smf.logit(formula, data=model_df).fit(disp=False)

# Control model excluding denied_PMI (possible mediator)
controls_nopmi = [c for c in controls if c != 'denied_PMI']
model_df_nopmi = df.dropna(subset=['accept','female'] + controls_nopmi)
formula_nopmi = 'accept ~ female + ' + ' + '.join(controls_nopmi)
logit_ctrl_nopmi = smf.logit(formula_nopmi, data=model_df_nopmi).fit(disp=False)

# Extract results
simple_coef = logit_simple.params['female']
simple_p = logit_simple.pvalues['female']
simple_or = np.exp(simple_coef)

ctrl_coef = logit_ctrl.params['female']
ctrl_p = logit_ctrl.pvalues['female']
ctrl_or = np.exp(ctrl_coef)

ctrl2_coef = logit_ctrl_nopmi.params['female']
ctrl2_p = logit_ctrl_nopmi.pvalues['female']
ctrl2_or = np.exp(ctrl2_coef)

# Save a small report for debugging
report = {
    'n': int(n),
    'accept_rate_overall': float(accept_rate),
    'accept_rate_female': float(accept_f),
    'accept_rate_male': float(accept_m),
    'chi2_p': float(p_chi),
    'diff_prop': float(p1 - p0),
    'diff_prop_p': float(p_diff),
    'simple_logit_coef': float(simple_coef),
    'simple_logit_or': float(simple_or),
    'simple_logit_p': float(simple_p),
    'ctrl_logit_coef': float(ctrl_coef),
    'ctrl_logit_or': float(ctrl_or),
    'ctrl_logit_p': float(ctrl_p),
    'n_model_ctrl': int(len(model_df)),
    'ctrl_nopmi_logit_coef': float(ctrl2_coef),
    'ctrl_nopmi_logit_or': float(ctrl2_or),
    'ctrl_nopmi_logit_p': float(ctrl2_p),
    'n_model_ctrl_nopmi': int(len(model_df_nopmi)),
}

pd.Series(report).to_json('analysis_report.json')
print(report)
