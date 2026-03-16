import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = 'teachingratings.csv'
df = pd.read_csv(csv_path)

# Basic inspection
print('Columns:', list(df.columns))
print('Head:')
print(df.head())
print('\nSummary:')
print(df.describe(include='all'))

# Outcome and predictor
outcome = 'allstudents'
predictor = 'beauty'

# Ensure numeric types
for col in [outcome, predictor, 'age', 'division', 'rownames', 'minority', 'students']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing outcome or predictor
model_df = df.dropna(subset=[outcome, predictor]).copy()

# Correlation
corr = model_df[[outcome, predictor]].corr().iloc[0,1]
print('\nPearson correlation (beauty vs allstudents):', corr)

# Simple OLS
simple_model = smf.ols(f'{outcome} ~ {predictor}', data=model_df).fit(cov_type='HC3')
print('\nSimple OLS (HC3):')
print(simple_model.summary())

# Controls: include categorical factors and numeric controls (excluding division as likely ID)
cat_cols = [c for c in ['eval', 'tenure', 'prof', 'native', 'gender', 'credits'] if c in model_df.columns]
num_cols = [c for c in ['age', 'rownames', 'minority', 'students'] if c in model_df.columns]

# Build formula with C() for categorical
terms = [predictor] + [f'C({c})' for c in cat_cols] + num_cols
formula = outcome + ' ~ ' + ' + '.join(terms)

controls_model = smf.ols(formula, data=model_df).fit(cov_type='HC3')
print('\nControls OLS (HC3):')
print(controls_model.summary())

# Effect size: predicted change for 1 SD increase in beauty
beauty_sd = model_df[predictor].std()
beauty_coef = simple_model.params[predictor]
beauty_coef_controls = controls_model.params.get(predictor, np.nan)
print('\nBeauty SD:', beauty_sd)
print('Simple model: coef', beauty_coef, '1 SD effect', beauty_coef * beauty_sd)
print('Controls model: coef', beauty_coef_controls, '1 SD effect', beauty_coef_controls * beauty_sd)

# Save key results to a small dataframe for later reference
results = {
    'corr': corr,
    'simple_coef': beauty_coef,
    'simple_p': simple_model.pvalues[predictor],
    'simple_ci_low': simple_model.conf_int().loc[predictor, 0],
    'simple_ci_high': simple_model.conf_int().loc[predictor, 1],
    'controls_coef': beauty_coef_controls,
    'controls_p': controls_model.pvalues.get(predictor, np.nan),
    'controls_ci_low': controls_model.conf_int().loc[predictor, 0],
    'controls_ci_high': controls_model.conf_int().loc[predictor, 1],
    'beauty_sd': beauty_sd,
}

print('\nKey results:')
for k,v in results.items():
    print(f'{k}: {v}')
