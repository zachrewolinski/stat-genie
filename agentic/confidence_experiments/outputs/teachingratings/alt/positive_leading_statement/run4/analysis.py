import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
csv_path = 'teachingratings.csv'
df = pd.read_csv(csv_path)

# Basic cleaning: ensure categorical columns are treated as categories
cat_cols = [
    'minority', 'gender', 'credits', 'division', 'native', 'tenure'
]
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Outcome and main predictor
outcome = 'eval'
predictor = 'beauty'

# Drop rows with missing values in used columns
base_cols = [outcome, predictor]
base_df = df[base_cols].dropna()

# Simple correlation
corr = base_df[outcome].corr(base_df[predictor])

# Simple bivariate regression
model_simple = smf.ols(f"{outcome} ~ {predictor}", data=base_df).fit(cov_type='HC3')

# Multivariate regression with controls
control_cols = ['age', 'gender', 'minority', 'native', 'tenure', 'division', 'credits', 'students']
available_controls = [c for c in control_cols if c in df.columns]

formula_controls = outcome + ' ~ ' + predictor
if available_controls:
    # Use C() for categorical variables
    terms = [predictor]
    for c in available_controls:
        if str(df[c].dtype) == 'category':
            terms.append(f'C({c})')
        else:
            terms.append(c)
    formula_controls = outcome + ' ~ ' + ' + '.join(terms)

control_df = df[[outcome, predictor] + available_controls].dropna()
model_controls = smf.ols(formula_controls, data=control_df).fit(cov_type='HC3')

# Compute standardized effect for beauty in controlled model
beauty_sd = control_df[predictor].std()
# For standardized effect on outcome in original units, multiply coefficient by 1 SD
coef_beauty = model_controls.params.get(predictor, np.nan)
std_effect = coef_beauty * beauty_sd

# Save key results for use in response
results = {
    'n_total': len(df),
    'n_simple': int(model_simple.nobs),
    'n_controls': int(model_controls.nobs),
    'corr': corr,
    'simple_coef': model_simple.params[predictor],
    'simple_p': model_simple.pvalues[predictor],
    'simple_ci': model_simple.conf_int().loc[predictor].tolist(),
    'controls_formula': formula_controls,
    'controls_coef': coef_beauty,
    'controls_p': model_controls.pvalues.get(predictor, np.nan),
    'controls_ci': model_controls.conf_int().loc[predictor].tolist(),
    'beauty_sd': beauty_sd,
    'std_effect_eval_units': std_effect,
    'eval_sd': control_df[outcome].std(),
    'model_controls_r2': model_controls.rsquared,
}

print(json.dumps(results, indent=2))
