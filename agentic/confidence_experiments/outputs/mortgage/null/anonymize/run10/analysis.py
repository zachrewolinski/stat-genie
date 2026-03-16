import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data

df = pd.read_csv('mortgage.csv')

# Define variables based on metadata
# feature2: female indicator
# feature14: accepted (1) / denied (0)

gender = 'feature2'
outcome = 'feature14'

# Approval rates by gender
base = df[[gender, outcome]].dropna()
rates = base.groupby(gender)[outcome].agg(['mean', 'count'])

# Contingency table and chi-square test
ct = pd.crosstab(base[gender], base[outcome])
chi2, p_chi, dof, exp = stats.chi2_contingency(ct)

# Logistic regression with controls
# Exclude feature1 (unique id). Exclude feature11 and feature14 (outcome)
control_cols = [
    'feature3',  # Black indicator
    'feature4',  # housing expense ratio
    'feature5',  # self-employed
    'feature6',  # married
    'feature7',  # mortgage credit score
    'feature8',  # consumer credit score
    'feature9',  # bad credit history
    'feature10', # debt-to-income
    'feature12', # loan-to-value
    'feature13', # PMI denied
]

model_df = df[[gender, outcome] + control_cols].dropna()
X = model_df[[gender] + control_cols]
X = sm.add_constant(X)
y = model_df[outcome]

logit = sm.Logit(y, X)
res = logit.fit(disp=False)

coef = float(res.params[gender])
se = float(res.bse[gender])
pval = float(res.pvalues[gender])

odds_ratio = float(np.exp(coef))
conf_int = res.conf_int().loc[gender]
conf_or = np.exp(conf_int)

results = {
    'n_total': int(len(df)),
    'n_used': int(len(model_df)),
    'approval_rates': {
        'male_0': {'mean': float(rates.loc[0, 'mean']), 'count': int(rates.loc[0, 'count'])},
        'female_1': {'mean': float(rates.loc[1, 'mean']), 'count': int(rates.loc[1, 'count'])},
    },
    'chi_square': {'chi2': float(chi2), 'p_value': float(p_chi), 'dof': int(dof)},
    'logit_gender': {
        'coef': coef,
        'se': se,
        'p_value': pval,
        'odds_ratio': odds_ratio,
        'or_ci_low': float(conf_or[0]),
        'or_ci_high': float(conf_or[1]),
    },
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
