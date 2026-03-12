import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('teachingratings.csv')

# Identify key columns
beauty_col = 'beauty'
rating_col = 'allstudents'

# Basic stats
print('n:', len(df))
print('missing beauty:', df[beauty_col].isna().sum())
print('missing rating:', df[rating_col].isna().sum())

# Correlation
corr = df[[beauty_col, rating_col]].corr().iloc[0,1]
print('pearson_corr:', corr)

# Simple OLS
model_simple = smf.ols(f"{rating_col} ~ {beauty_col}", data=df).fit()
print(model_simple.summary())

# Multiple regression with controls
# Choose potential controls (exclude IDs)
controls = ['age', 'tenure', 'prof', 'native', 'gender', 'credits', 'rownames', 'minority', 'students']

# Ensure controls exist
controls = [c for c in controls if c in df.columns]

# Build formula with categorical controls
cat_controls = ['tenure', 'prof', 'native', 'gender', 'credits']

control_terms = []
for c in controls:
    if c in cat_controls:
        control_terms.append(f"C({c})")
    else:
        control_terms.append(c)

formula = f"{rating_col} ~ {beauty_col} + " + " + ".join(control_terms)
model_multi = smf.ols(formula, data=df).fit()
print(model_multi.summary())

# Extract key stats
simple_coef = model_simple.params[beauty_col]
simple_p = model_simple.pvalues[beauty_col]

multi_coef = model_multi.params[beauty_col]
multi_p = model_multi.pvalues[beauty_col]

print('simple_coef', simple_coef, 'simple_p', simple_p)
print('multi_coef', multi_coef, 'multi_p', multi_p)

