import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Variables of interest
vars_basic = ['eval', 'beauty']

# Basic correlation
basic_df = df[vars_basic].dropna()
pearson_r, pearson_p = stats.pearsonr(basic_df['beauty'], basic_df['eval'])

# Simple OLS
model_simple = smf.ols('eval ~ beauty', data=basic_df).fit()

# Controls (chosen based on dataset context)
controls = ['age', 'students', 'allstudents', 'C(gender)', 'C(minority)', 'C(native)', 'C(tenure)', 'C(division)', 'C(credits)']
vars_controls = ['eval', 'beauty', 'age', 'students', 'allstudents', 'gender', 'minority', 'native', 'tenure', 'division', 'credits']
control_df = df[vars_controls].dropna()

formula_controls = 'eval ~ beauty + age + students + allstudents + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits)'
model_controls = smf.ols(formula_controls, data=control_df).fit()

# Standardized effect for beauty in controlled model (beta)
# Standardize beauty and eval (and numeric controls) for beta estimate
std_df = control_df.copy()
# Standardize numeric columns
for col in ['eval', 'beauty', 'age', 'students', 'allstudents']:
    std_df[col] = (std_df[col] - std_df[col].mean()) / std_df[col].std(ddof=0)
model_controls_std = smf.ols(formula_controls, data=std_df).fit()

results = {
    'n_total': len(df),
    'n_basic': len(basic_df),
    'n_controls': len(control_df),
    'pearson_r': pearson_r,
    'pearson_p': pearson_p,
    'simple_coef': model_simple.params['beauty'],
    'simple_p': model_simple.pvalues['beauty'],
    'simple_r2': model_simple.rsquared,
    'controls_coef': model_controls.params['beauty'],
    'controls_p': model_controls.pvalues['beauty'],
    'controls_r2': model_controls.rsquared,
    'controls_beta_std': model_controls_std.params['beauty'],
}

print(results)
