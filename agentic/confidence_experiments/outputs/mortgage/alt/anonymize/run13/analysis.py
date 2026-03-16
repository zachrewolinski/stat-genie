import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
df = pd.read_csv('mortgage.csv')
df = df.replace([np.inf, -np.inf], np.nan)

# Identify columns
# feature2 = gender (1 female, 0 male)
# feature14 = accepted (1 accepted, 0 denied)

unadj_data = df[['feature2', 'feature14']].dropna()
gender = unadj_data['feature2']
accepted = unadj_data['feature14']

# Basic counts
ct = pd.crosstab(gender, accepted)
# Ensure columns 0/1 exist
ct = ct.reindex(index=[0, 1], columns=[0, 1], fill_value=0)

# Approval rates
rate_male = ct.loc[0,1] / ct.loc[0].sum() if ct.loc[0].sum() else np.nan
rate_female = ct.loc[1,1] / ct.loc[1].sum() if ct.loc[1].sum() else np.nan

# Chi-square test
chi2, p_chi, dof, expected = stats.chi2_contingency(ct.values)

# Logistic regression: accepted ~ gender (unadjusted)
X_unadj = sm.add_constant(unadj_data['feature2'])
model_unadj = sm.Logit(unadj_data['feature14'], X_unadj)
res_unadj = model_unadj.fit(disp=False)

# Adjusted model: accepted ~ gender + controls (exclude outcomes)
control_cols = [
    'feature3','feature4','feature5','feature6','feature7','feature8',
    'feature9','feature10','feature12','feature13'
]
adj_data = df[['feature2', 'feature14'] + control_cols].dropna()
X_adj = adj_data[['feature2'] + control_cols]
X_adj = sm.add_constant(X_adj)
model_adj = sm.Logit(adj_data['feature14'], X_adj)
res_adj = model_adj.fit(disp=False)

# Extract effect of gender (feature2)
coef_unadj = res_unadj.params['feature2']
se_unadj = res_unadj.bse['feature2']
p_unadj = res_unadj.pvalues['feature2']

coef_adj = res_adj.params['feature2']
se_adj = res_adj.bse['feature2']
p_adj = res_adj.pvalues['feature2']

# Odds ratios
or_unadj = np.exp(coef_unadj)
or_adj = np.exp(coef_adj)

# Confidence intervals for odds ratios
ci_unadj = res_unadj.conf_int().loc['feature2']
ci_adj = res_adj.conf_int().loc['feature2']
or_ci_unadj = (float(np.exp(ci_unadj[0])), float(np.exp(ci_unadj[1])))
or_ci_adj = (float(np.exp(ci_adj[0])), float(np.exp(ci_adj[1])))

# Simple risk difference
risk_diff = rate_female - rate_male

summary = {
    'n_total': int(len(df)),
    'n_unadj': int(len(unadj_data)),
    'n_adj': int(len(adj_data)),
    'ct': ct.to_dict(),
    'rate_male': float(rate_male),
    'rate_female': float(rate_female),
    'risk_diff': float(risk_diff),
    'chi2_p': float(p_chi),
    'unadj': {
        'coef': float(coef_unadj),
        'se': float(se_unadj),
        'p': float(p_unadj),
        'or': float(or_unadj),
        'or_ci': or_ci_unadj
    },
    'adj': {
        'coef': float(coef_adj),
        'se': float(se_adj),
        'p': float(p_adj),
        'or': float(or_adj),
        'or_ci': or_ci_adj
    }
}

print(summary)
