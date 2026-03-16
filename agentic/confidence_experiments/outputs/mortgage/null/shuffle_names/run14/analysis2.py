import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2_contingency

# Load data
path = 'mortgage.csv'
df = pd.read_csv(path)

# Identify columns based on metadata descriptions
female_col = 'denied_PMI'  # description says 1 if applicant is female
approval_col = 'deny'      # description says 1 if application accepted

# Drop rows with missing values in key columns
base_df = df[[female_col, approval_col]].copy()
base_df = base_df.dropna()

# Basic rates
approval_rate = base_df[approval_col].mean()
approval_by_gender = base_df.groupby(female_col)[approval_col].mean()
count_by_gender = base_df[female_col].value_counts().sort_index()

# 2x2 contingency table for chi-square
ct = pd.crosstab(base_df[female_col], base_df[approval_col])
chi2, p, dof, expected = chi2_contingency(ct)

# Unadjusted logistic regression: approval ~ female
X = sm.add_constant(base_df[[female_col]])
model = sm.Logit(base_df[approval_col], X).fit(disp=False)

# Adjusted logistic regression with other covariates (all other columns)
# Exclude any non-numeric or identifier columns if needed
covariates = [c for c in df.columns if c not in {approval_col, female_col}]
X_adj = df[covariates].apply(pd.to_numeric, errors='coerce')

adj_df = pd.concat([df[[approval_col, female_col]], X_adj], axis=1).dropna()
X_adj = sm.add_constant(adj_df[[female_col] + covariates])
model_adj = sm.Logit(adj_df[approval_col], X_adj).fit(disp=False)

# Extract effect sizes
odds_ratio = np.exp(model.params[female_col])
odds_ratio_adj = np.exp(model_adj.params[female_col])

# Print key results
print('Approval rate overall:', approval_rate)
print('Approval rate by gender (0=male,1=female)')
print(approval_by_gender)
print('Counts by gender')
print(count_by_gender)
print('Chi-square p-value:', p)
print('Unadjusted logit coef (female):', model.params[female_col])
print('Unadjusted logit p-value (female):', model.pvalues[female_col])
print('Unadjusted odds ratio (female vs male):', odds_ratio)
print('Adjusted logit coef (female):', model_adj.params[female_col])
print('Adjusted logit p-value (female):', model_adj.pvalues[female_col])
print('Adjusted odds ratio (female vs male):', odds_ratio_adj)

