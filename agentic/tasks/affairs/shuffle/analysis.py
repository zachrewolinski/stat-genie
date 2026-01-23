"""
Analysis of Affairs Dataset: Children vs Extramarital Affairs
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import json

# Load data
df = pd.read_csv('affairs.csv')

print("=" * 80)
print("AFFAIRS DATASET ANALYSIS: Children vs Extramarital Affairs")
print("=" * 80)

# Load metadata to identify columns
with open('info.json', 'r') as f:
    info = json.load(f)

print("\n1. IDENTIFYING COLUMNS FROM SHUFFLED DATA")
print("-" * 80)

# The column names are shuffled. We need to identify columns by their properties.
# Let's check each column's properties to match with info.json metadata

def identify_column(df, col_name, expected_props):
    """Identify if a column matches expected properties"""
    col = df[col_name]

    # Check dtype
    if expected_props.get('dtype') == 'category':
        # Should be string-like
        if col.dtype == 'object':
            unique_vals = set(col.unique())
            expected_samples = set(expected_props.get('samples', []))
            if unique_vals == expected_samples:
                return True
    elif expected_props.get('dtype') == 'number':
        # Should be numeric
        if pd.api.types.is_numeric_dtype(col):
            col_min = col.min()
            col_max = col.max()
            exp_min = expected_props.get('min')
            exp_max = expected_props.get('max')
            # Allow small tolerance for float comparison
            if abs(col_min - exp_min) < 0.01 and abs(col_max - exp_max) < 0.01:
                return True
    return False

# Find the correct column for each variable
column_mapping = {}
for field in info['data_desc']['fields']:
    field_name = field['column']
    props = field['properties']

    # Try to find matching column
    for col in df.columns:
        if identify_column(df, col, props):
            column_mapping[field_name] = col
            print(f"  {field_name:15s} -> actual column: {col}")
            break

# Rename columns to their true names
df_renamed = df.rename(columns={v: k for k, v in column_mapping.items()})

# Verify we have the key columns
assert 'affairs' in df_renamed.columns, "Could not identify affairs column"
assert 'children' in df_renamed.columns, "Could not identify children column"

print("\n2. DESCRIPTIVE STATISTICS")
print("-" * 80)

# Sample size
n_total = len(df_renamed)
print(f"Total sample size: {n_total}")

# Affairs distribution
print(f"\nAffairs outcome (past year frequency):")
print(f"  Mean: {df_renamed['affairs'].mean():.3f}")
print(f"  Median: {df_renamed['affairs'].median():.3f}")
print(f"  Std Dev: {df_renamed['affairs'].std():.3f}")
print(f"  Min: {df_renamed['affairs'].min():.1f}")
print(f"  Max: {df_renamed['affairs'].max():.1f}")

# Proportion of zeros
n_zero = (df_renamed['affairs'] == 0).sum()
pct_zero = 100 * n_zero / n_total
print(f"  Zero affairs: {n_zero} ({pct_zero:.1f}%)")
print(f"  Any affairs: {n_total - n_zero} ({100 - pct_zero:.1f}%)")

# Value distribution
print(f"\nAffairs value distribution:")
affairs_counts = df_renamed['affairs'].value_counts().sort_index()
for val, count in affairs_counts.items():
    pct = 100 * count / n_total
    print(f"  {val:4.0f}: {count:3d} ({pct:5.1f}%)")

print("\n3. CHILDREN VARIABLE")
print("-" * 80)

# Children distribution
children_counts = df_renamed['children'].value_counts()
print(f"Children in marriage:")
for val, count in children_counts.items():
    pct = 100 * count / n_total
    print(f"  {val}: {count} ({pct:.1f}%)")

print("\n4. AFFAIRS BY CHILDREN STATUS")
print("-" * 80)

# Group by children
grouped = df_renamed.groupby('children')['affairs']

print(f"Sample sizes:")
for name, group in grouped:
    print(f"  {name}: n = {len(group)}")

print(f"\nMean affairs by children status:")
means = grouped.mean()
for name, val in means.items():
    print(f"  {name}: {val:.3f}")

print(f"\nMedian affairs by children status:")
medians = grouped.median()
for name, val in medians.items():
    print(f"  {name}: {val:.3f}")

print(f"\nProportion with any affairs (>0):")
for name, group in grouped:
    n_any = (group > 0).sum()
    pct = 100 * n_any / len(group)
    print(f"  {name}: {n_any}/{len(group)} ({pct:.1f}%)")

# Statistical test for difference in means
children_yes = df_renamed[df_renamed['children'] == 'yes']['affairs']
children_no = df_renamed[df_renamed['children'] == 'no']['affairs']
t_stat, p_value = stats.ttest_ind(children_yes, children_no)
print(f"\nT-test for difference in means:")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_value:.4f}")
diff_means = children_yes.mean() - children_no.mean()
print(f"  Difference (yes - no): {diff_means:.3f}")

print("\n5. MODEL A: SIMPLE BIVARIATE ASSOCIATION")
print("-" * 80)

# Create binary indicator for children
df_renamed['children_yes'] = (df_renamed['children'] == 'yes').astype(int)

# Simple OLS regression
X_simple = sm.add_constant(df_renamed['children_yes'])
y = df_renamed['affairs']

model_a = OLS(y, X_simple).fit()
print(f"\nOLS: affairs ~ children")
print(f"  Coefficient (children=yes): {model_a.params['children_yes']:.4f}")
print(f"  Std Error: {model_a.bse['children_yes']:.4f}")
print(f"  t-statistic: {model_a.tvalues['children_yes']:.3f}")
print(f"  p-value: {model_a.pvalues['children_yes']:.4f}")
print(f"  95% CI: [{model_a.conf_int().loc['children_yes', 0]:.4f}, {model_a.conf_int().loc['children_yes', 1]:.4f}]")
print(f"  R-squared: {model_a.rsquared:.4f}")

# Interpretation
print(f"\nInterpretation:")
print(f"  Having children is associated with {model_a.params['children_yes']:.3f} {'fewer' if model_a.params['children_yes'] < 0 else 'more'} affairs")
print(f"  on the coded scale (on average).")

print("\n6. MODEL B: MULTIVARIABLE MODEL WITH COVARIATES")
print("-" * 80)

# Prepare covariates
# Create dummy for gender if it exists
if 'gender' in df_renamed.columns:
    df_renamed['gender_male'] = (df_renamed['gender'] == 'male').astype(int)
    gender_vars = ['gender_male']
else:
    gender_vars = []

# Control variables: age, yearsmarried, religiousness, rating, education
control_vars = []
for var in ['age', 'yearsmarried', 'religiousness', 'rating', 'education', 'occupation']:
    if var in df_renamed.columns:
        control_vars.append(var)

print(f"Control variables: {', '.join(control_vars + gender_vars)}")

# Build feature matrix
X_multi = df_renamed[['children_yes'] + control_vars + gender_vars].copy()
X_multi = sm.add_constant(X_multi)

model_b = OLS(y, X_multi).fit()
print(f"\nOLS with covariates: affairs ~ children + controls")
print(f"  Coefficient (children=yes): {model_b.params['children_yes']:.4f}")
print(f"  Std Error: {model_b.bse['children_yes']:.4f}")
print(f"  t-statistic: {model_b.tvalues['children_yes']:.3f}")
print(f"  p-value: {model_b.pvalues['children_yes']:.4f}")
print(f"  95% CI: [{model_b.conf_int().loc['children_yes', 0]:.4f}, {model_b.conf_int().loc['children_yes', 1]:.4f}]")
print(f"  R-squared: {model_b.rsquared:.4f}")

print(f"\nKey control variable coefficients:")
for var in ['rating', 'religiousness', 'yearsmarried']:
    if var in model_b.params.index:
        print(f"  {var:15s}: {model_b.params[var]:8.4f} (p={model_b.pvalues[var]:.4f})")

print("\n7. ALTERNATIVE MODEL: LOGISTIC (ANY AFFAIRS)")
print("-" * 80)

# Binary outcome: any affairs
df_renamed['any_affairs'] = (df_renamed['affairs'] > 0).astype(int)

# Logistic regression
from statsmodels.discrete.discrete_model import Logit

X_logit = df_renamed[['children_yes'] + control_vars + gender_vars].copy()
X_logit = sm.add_constant(X_logit)
y_binary = df_renamed['any_affairs']

model_logit = Logit(y_binary, X_logit).fit(disp=0)
print(f"\nLogistic regression: P(any affairs) ~ children + controls")
print(f"  Coefficient (children=yes): {model_logit.params['children_yes']:.4f}")
print(f"  Std Error: {model_logit.bse['children_yes']:.4f}")
print(f"  z-statistic: {model_logit.tvalues['children_yes']:.3f}")
print(f"  p-value: {model_logit.pvalues['children_yes']:.4f}")
print(f"  Odds Ratio: {np.exp(model_logit.params['children_yes']):.4f}")

# Compute marginal effect
# Average marginal effect
probs = model_logit.predict(X_logit)
avg_slope = model_logit.params['children_yes'] * np.mean(probs * (1 - probs))
print(f"  Average marginal effect: {avg_slope:.4f}")
print(f"    (children associated with {abs(avg_slope)*100:.2f} percentage point {'decrease' if avg_slope < 0 else 'increase'} in probability)")

print("\n8. DIAGNOSTICS AND SANITY CHECKS")
print("-" * 80)

# Check for missing values
print(f"Missing values:")
missing = df_renamed[['affairs', 'children'] + control_vars + gender_vars].isnull().sum()
if missing.sum() == 0:
    print(f"  None detected")
else:
    print(missing[missing > 0])

# Check correlation between children and key confounders
print(f"\nCorrelation between children_yes and controls:")
for var in ['yearsmarried', 'age', 'rating']:
    if var in df_renamed.columns:
        corr = df_renamed['children_yes'].corr(df_renamed[var])
        print(f"  {var:15s}: {corr:.3f}")

# Distribution check
print(f"\nModel residual checks (Model B):")
residuals = model_b.resid
print(f"  Mean residual: {residuals.mean():.6f} (should be ~0)")
print(f"  Std residual: {residuals.std():.3f}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nKEY FINDINGS SUMMARY:")
print(f"  - Sample size: {n_total}")
print(f"  - Baseline: {pct_zero:.1f}% had no affairs")
print(f"  - Simple association (Model A): {model_a.params['children_yes']:.3f} (p={model_a.pvalues['children_yes']:.4f})")
print(f"  - Adjusted association (Model B): {model_b.params['children_yes']:.3f} (p={model_b.pvalues['children_yes']:.4f})")
print(f"  - Logistic marginal effect: {avg_slope*100:.2f} percentage points")
