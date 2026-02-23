import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load dataset
df = pd.read_csv('affairs.csv')

# Reconstruct semantics based on info.json descriptions:
# - 'age' column encodes frequency of extramarital intercourse in the past year.
# - 'religiousness' column is actually a yes/no indicator for whether there are children in the marriage.
# - Other columns are potential covariates but the core research question is about children.

# Outcome: any extramarital affair (binary)
df['any_affair'] = (df['age'] > 0).astype(int)

# Predictor: having children (1 = yes, 0 = no)
df['has_children'] = (df['religiousness'].str.lower() == 'yes').astype(int)

# Basic descriptive statistics: proportion with any affair by children status
summary = (
    df.groupby('has_children')['any_affair']
    .agg(['mean', 'count', 'sum'])
    .rename(columns={'mean': 'prop_any_affair', 'sum': 'num_with_affair'})
)
print("Proportion with any extramarital affair by children status (has_children=1 means yes):")
print(summary)
print()

# Simple logistic regression: any_affair ~ has_children
X = sm.add_constant(df['has_children'])
model_simple = sm.Logit(df['any_affair'], X).fit(disp=False)
print("Simple logistic regression (any_affair ~ has_children):")
print(model_simple.summary())

# Extract key statistics
coef_children = model_simple.params['has_children']
se_children = model_simple.bse['has_children']
z_children = coef_children / se_children
p_children = model_simple.pvalues['has_children']
odds_ratio_children = float(np.exp(coef_children))

print("\nCoefficient for has_children:", coef_children)
print("Standard error:", se_children)
print("z-statistic:", z_children)
print("p-value:", p_children)
print("Odds ratio (has_children vs no children):", odds_ratio_children)

# Also report mean affair frequency (original scale) by children status for additional context
freq_summary = df.groupby('has_children')['age'].agg(['mean', 'std', 'count'])
print("\nMean affair frequency (age column, 0-12 scale) by children status:")
print(freq_summary)
