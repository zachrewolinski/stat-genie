import pandas as pd
import numpy as np
import json
from scipy import stats
from scipy.stats import ttest_ind
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.discrete.discrete_model import Poisson

# Load data
df = pd.read_csv('affairs.csv')

# Load metadata to understand column mapping
with open('info.json', 'r') as f:
    info = json.load(f)

# Map feature names to semantic names based on info.json field_names
field_names = info['data_desc']['field_names']
col_mapping = {f'feature{i+1}': field_names[i] for i in range(len(field_names))}

# Rename columns for clarity
df_renamed = df.rename(columns=col_mapping)

print("="*70)
print("AFFAIRS DATASET ANALYSIS: Children and Extramarital Affairs")
print("="*70)

# Identify key variables
outcome_var = 'affairs'
key_var = 'children'

print(f"\nColumn mapping identified from info.json:")
print(f"  Outcome variable: {outcome_var}")
print(f"  Key explanatory variable: {key_var}")
print(f"  Sample size: {len(df_renamed)}")

# Basic descriptive statistics
print("\n" + "="*70)
print("DESCRIPTIVE STATISTICS")
print("="*70)

print(f"\nAffairs distribution:")
print(df_renamed[outcome_var].describe())
print(f"\nProportion with zero affairs: {(df_renamed[outcome_var] == 0).mean():.3f}")
print(f"Proportion with any affairs: {(df_renamed[outcome_var] > 0).mean():.3f}")

# Children groups
print(f"\n{key_var.capitalize()} distribution:")
print(df_renamed[key_var].value_counts())

# Group statistics by children
print(f"\nAffairs by {key_var} status:")
grouped_stats = df_renamed.groupby(key_var)[outcome_var].agg([
    ('n', 'count'),
    ('mean', 'mean'),
    ('median', 'median'),
    ('std', 'std'),
    ('prop_any_affair', lambda x: (x > 0).mean())
])
print(grouped_stats)

# Statistical test of means
no_children = df_renamed[df_renamed[key_var] == 'no'][outcome_var]
yes_children = df_renamed[df_renamed[key_var] == 'yes'][outcome_var]
t_stat, p_value = ttest_ind(no_children, yes_children)
print(f"\nT-test comparing means:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Mean difference (no - yes): {no_children.mean() - yes_children.mean():.4f}")

# Additional descriptive stats for covariates
print(f"\nKey covariates summary:")
for var in ['age', 'yearsmarried', 'religiousness', 'rating', 'education']:
    if var in df_renamed.columns:
        print(f"  {var}: mean={df_renamed[var].mean():.2f}, std={df_renamed[var].std():.2f}")

print("\n" + "="*70)
print("MODEL A: SIMPLE ASSOCIATION (CHILDREN ~ AFFAIRS)")
print("="*70)

# Model A: Simple linear regression
# Create binary indicator for children
df_renamed['children_yes'] = (df_renamed[key_var] == 'yes').astype(int)

X_simple = sm.add_constant(df_renamed['children_yes'])
y = df_renamed[outcome_var]

model_a = OLS(y, X_simple).fit()
print("\nOLS Regression Results:")
print(model_a.summary().tables[1])

print(f"\nModel A Interpretation:")
print(f"  Intercept (no children): {model_a.params['const']:.4f}")
print(f"  Coefficient for children: {model_a.params['children_yes']:.4f}")
print(f"  95% CI for children effect: [{model_a.conf_int().loc['children_yes', 0]:.4f}, {model_a.conf_int().loc['children_yes', 1]:.4f}]")
print(f"  P-value: {model_a.pvalues['children_yes']:.4f}")

print("\n" + "="*70)
print("MODEL B: MULTIVARIABLE MODEL (CONTROLLING FOR CONFOUNDERS)")
print("="*70)

# Model B: Multiple regression controlling for confounders
# Convert categorical variables
df_renamed['gender_male'] = (df_renamed['gender'] == 'male').astype(int)

# Select control variables that might confound the relationship
control_vars = ['age', 'yearsmarried', 'religiousness', 'rating', 'education', 'gender_male']
available_controls = [var for var in control_vars if var in df_renamed.columns or var == 'gender_male']

X_multi = df_renamed[['children_yes'] + available_controls].copy()
X_multi = sm.add_constant(X_multi)

model_b = OLS(y, X_multi).fit()
print("\nMultivariable OLS Results:")
print(model_b.summary().tables[1])

print(f"\nModel B Interpretation (adjusted for confounders):")
print(f"  Coefficient for children: {model_b.params['children_yes']:.4f}")
print(f"  95% CI for children effect: [{model_b.conf_int().loc['children_yes', 0]:.4f}, {model_b.conf_int().loc['children_yes', 1]:.4f}]")
print(f"  P-value: {model_b.pvalues['children_yes']:.4f}")
print(f"  R-squared: {model_b.rsquared:.4f}")

print("\n" + "="*70)
print("ROBUSTNESS CHECK: POISSON MODEL")
print("="*70)
print("(Appropriate for non-negative count data)")

# Poisson model as robustness check
# Since affairs has many zeros and is non-negative, Poisson might be appropriate
model_poisson = Poisson(y, X_multi).fit(maxiter=100, disp=False)
print("\nPoisson Regression Results:")
print(model_poisson.summary().tables[1])

print(f"\nPoisson Model Interpretation:")
print(f"  Coefficient for children: {model_poisson.params['children_yes']:.4f}")
print(f"  Incident Rate Ratio: {np.exp(model_poisson.params['children_yes']):.4f}")
print(f"  95% CI for IRR: [{np.exp(model_poisson.conf_int().loc['children_yes', 0]):.4f}, {np.exp(model_poisson.conf_int().loc['children_yes', 1]):.4f}]")
print(f"  P-value: {model_poisson.pvalues['children_yes']:.4f}")

print("\n" + "="*70)
print("LOGISTIC REGRESSION: ANY AFFAIRS (YES/NO)")
print("="*70)

# Binary outcome: any affairs vs none
df_renamed['any_affairs'] = (df_renamed[outcome_var] > 0).astype(int)

from statsmodels.discrete.discrete_model import Logit
model_logit = Logit(df_renamed['any_affairs'], X_multi).fit(disp=False)
print("\nLogistic Regression Results:")
print(model_logit.summary().tables[1])

print(f"\nLogistic Model Interpretation:")
print(f"  Coefficient for children: {model_logit.params['children_yes']:.4f}")
print(f"  Odds Ratio: {np.exp(model_logit.params['children_yes']):.4f}")
print(f"  95% CI for OR: [{np.exp(model_logit.conf_int().loc['children_yes', 0]):.4f}, {np.exp(model_logit.conf_int().loc['children_yes', 1]):.4f}]")
print(f"  P-value: {model_logit.pvalues['children_yes']:.4f}")

print("\n" + "="*70)
print("SUMMARY OF KEY FINDINGS")
print("="*70)

print(f"""
Sample sizes:
  - No children: {(df_renamed[key_var] == 'no').sum()}
  - Yes children: {(df_renamed[key_var] == 'yes').sum()}

Mean affairs frequency:
  - No children: {no_children.mean():.3f}
  - Yes children: {yes_children.mean():.3f}
  - Difference: {no_children.mean() - yes_children.mean():.3f}

Proportion with any affairs:
  - No children: {(no_children > 0).mean():.3f}
  - Yes children: {(yes_children > 0).mean():.3f}

Model results (children effect):
  - Simple model (A): β = {model_a.params['children_yes']:.4f}, p = {model_a.pvalues['children_yes']:.4f}
  - Adjusted model (B): β = {model_b.params['children_yes']:.4f}, p = {model_b.pvalues['children_yes']:.4f}
  - Poisson (IRR): {np.exp(model_poisson.params['children_yes']):.4f}, p = {model_poisson.pvalues['children_yes']:.4f}
  - Logistic (OR): {np.exp(model_logit.params['children_yes']):.4f}, p = {model_logit.pvalues['children_yes']:.4f}
""")

print("="*70)
print("Analysis complete. All numbers computed from data.")
print("="*70)
