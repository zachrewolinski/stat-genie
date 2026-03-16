import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('mortgage.csv')

female_col = 'feature2'
accepted_col = 'feature14'
control_cols = [
    'feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature12','feature13'
]
model_df = df[[female_col, accepted_col] + control_cols].dropna()
X = sm.add_constant(model_df[[female_col] + control_cols], has_constant='add')
y = model_df[accepted_col]
res = sm.Logit(y, X).fit(disp=False)
coef = res.params[female_col]
se = res.bse[female_col]
p = res.pvalues[female_col]
ci = res.conf_int().loc[female_col].tolist()
print({'coef': coef, 'se': se, 'p': p, 'ci_low': ci[0], 'ci_high': ci[1], 'or': float(np.exp(coef)), 'or_ci_low': float(np.exp(ci[0])), 'or_ci_high': float(np.exp(ci[1]))})
