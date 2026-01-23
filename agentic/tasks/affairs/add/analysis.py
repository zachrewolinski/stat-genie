#!/usr/bin/env python3
"""
Analysis of Affairs Dataset: Children and Extramarital Affairs
Research Question: Does having children decrease engagement in extramarital affairs?
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# Check if Tobit is available
try:
    from statsmodels.miscmodels.censored_model import Tobit
    TOBIT_AVAILABLE = True
except ImportError:
    TOBIT_AVAILABLE = False

# Load data
print("=" * 80)
print("AFFAIRS DATASET ANALYSIS: CHILDREN AND EXTRAMARITAL AFFAIRS")
print("=" * 80)
print()

df = pd.read_csv('affairs.csv')

# Select relevant columns based on info.json metadata
relevant_cols = ['affairs', 'gender', 'age', 'yearsmarried', 'children',
                 'religiousness', 'education', 'occupation', 'rating']
df = df[relevant_cols].copy()

# Clean data - remove any rows with missing values in key variables
df = df.dropna(subset=['affairs', 'children'])
print(f"Sample size: {len(df)} observations")
print()

# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================
print("DESCRIPTIVE STATISTICS")
print("-" * 80)

# Overall distribution of affairs
print(f"\nAffairs variable distribution:")
print(f"  Mean: {df['affairs'].mean():.3f}")
print(f"  Median: {df['affairs'].median():.3f}")
print(f"  Std Dev: {df['affairs'].std():.3f}")
print(f"  Min: {df['affairs'].min():.1f}, Max: {df['affairs'].max():.1f}")
print(f"  Proportion with zero affairs: {(df['affairs'] == 0).mean():.3f} ({(df['affairs'] == 0).sum()} obs)")
print(f"  Proportion with any affairs: {(df['affairs'] > 0).mean():.3f} ({(df['affairs'] > 0).sum()} obs)")

# Group by children status
print(f"\nSample sizes by children status:")
children_counts = df['children'].value_counts()
for status in ['no', 'yes']:
    if status in children_counts.index:
        count = children_counts[status]
        pct = count / len(df) * 100
        print(f"  Children = {status}: {count} ({pct:.1f}%)")

# Affairs by children status
print(f"\nAffairs by children status:")
for status in ['no', 'yes']:
    subset = df[df['children'] == status]
    if len(subset) > 0:
        print(f"  Children = {status}:")
        print(f"    Mean affairs: {subset['affairs'].mean():.3f}")
        print(f"    Median affairs: {subset['affairs'].median():.3f}")
        print(f"    Proportion with any affairs: {(subset['affairs'] > 0).mean():.3f}")

# Difference in means
mean_no_children = df[df['children'] == 'no']['affairs'].mean()
mean_yes_children = df[df['children'] == 'yes']['affairs'].mean()
diff_means = mean_no_children - mean_yes_children
print(f"\nRaw difference in mean affairs (no children - yes children): {diff_means:.3f}")

# T-test for difference
no_children = df[df['children'] == 'no']['affairs']
yes_children = df[df['children'] == 'yes']['affairs']
t_stat, p_val = stats.ttest_ind(no_children, yes_children)
print(f"T-test: t = {t_stat:.3f}, p = {p_val:.4f}")

# ============================================================================
# MODEL A: SIMPLE BASELINE ASSOCIATION
# ============================================================================
print("\n" + "=" * 80)
print("MODEL A: SIMPLE BASELINE (OLS)")
print("-" * 80)

# Create binary indicator for children (1 = yes, 0 = no)
df['children_yes'] = (df['children'] == 'yes').astype(int)

# Simple OLS regression
X_simple = sm.add_constant(df['children_yes'])
y = df['affairs']
model_a = sm.OLS(y, X_simple).fit()

print("\nOLS: affairs ~ children")
print(f"  Intercept: {model_a.params[0]:.3f} (SE: {model_a.bse[0]:.3f})")
print(f"  Children coefficient: {model_a.params[1]:.3f} (SE: {model_a.bse[1]:.3f})")
print(f"  95% CI for children: [{model_a.conf_int().loc['children_yes', 0]:.3f}, {model_a.conf_int().loc['children_yes', 1]:.3f}]")
print(f"  p-value: {model_a.pvalues[1]:.4f}")
print(f"  R-squared: {model_a.rsquared:.4f}")

# ============================================================================
# MODEL B: MULTIVARIABLE MODEL WITH CONTROLS
# ============================================================================
print("\n" + "=" * 80)
print("MODEL B: MULTIVARIABLE MODEL (OLS with controls)")
print("-" * 80)

# Create dummy for gender
df['gender_male'] = (df['gender'] == 'male').astype(int)

# Multivariable OLS with important confounders
# Include: age, years married, gender, religiousness, education, occupation, rating
X_multi = df[['children_yes', 'age', 'yearsmarried', 'gender_male',
              'religiousness', 'education', 'occupation', 'rating']].copy()
X_multi = sm.add_constant(X_multi)

model_b = sm.OLS(y, X_multi).fit()

print("\nOLS: affairs ~ children + age + yearsmarried + gender + religiousness + education + occupation + rating")
print(f"\nKey coefficient (children):")
print(f"  Children coefficient: {model_b.params['children_yes']:.3f} (SE: {model_b.bse['children_yes']:.3f})")
print(f"  95% CI: [{model_b.conf_int().loc['children_yes', 0]:.3f}, {model_b.conf_int().loc['children_yes', 1]:.3f}]")
print(f"  p-value: {model_b.pvalues['children_yes']:.4f}")
print(f"\nOther significant predictors:")
for var in ['age', 'yearsmarried', 'gender_male', 'religiousness', 'rating']:
    coef = model_b.params[var]
    pval = model_b.pvalues[var]
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    print(f"  {var}: {coef:.3f} (p={pval:.4f}) {sig}")
print(f"\nModel R-squared: {model_b.rsquared:.4f}")

# ============================================================================
# MODEL C: TOBIT MODEL (following Fair 1978)
# ============================================================================
if TOBIT_AVAILABLE:
    print("\n" + "=" * 80)
    print("MODEL C: TOBIT MODEL (censored at 0)")
    print("-" * 80)

    # Tobit model censored at 0 (following Fair's original approach)
    tobit_formula = 'affairs ~ children_yes + age + yearsmarried + gender_male + religiousness + education + occupation + rating'
    model_c = Tobit(y, X_multi).fit()

    print("\nTobit model (left-censored at 0)")
    print(f"\nKey coefficient (children):")
    print(f"  Children coefficient: {model_c.params['children_yes']:.3f} (SE: {model_c.bse['children_yes']:.3f})")
    print(f"  95% CI: [{model_c.conf_int().loc['children_yes', 0]:.3f}, {model_c.conf_int().loc['children_yes', 1]:.3f}]")
    print(f"  p-value: {model_c.pvalues['children_yes']:.4f}")
else:
    print("\n" + "=" * 80)
    print("MODEL C: TOBIT MODEL (not available in this statsmodels version)")
    print("-" * 80)
    print("Tobit model skipped - continuing with alternative approaches")

# ============================================================================
# ROBUSTNESS: LOG-TRANSFORMED OUTCOME
# ============================================================================
print("\n" + "=" * 80)
print("ROBUSTNESS CHECK: LOG-TRANSFORMED OUTCOME")
print("-" * 80)

# Log transformation: log(affairs + 1) to handle zeros
df['log_affairs'] = np.log(df['affairs'] + 1)

model_log = sm.OLS(df['log_affairs'], X_multi).fit()

print("\nOLS: log(affairs + 1) ~ children + controls")
print(f"  Children coefficient: {model_log.params['children_yes']:.3f} (SE: {model_log.bse['children_yes']:.3f})")
print(f"  95% CI: [{model_log.conf_int().loc['children_yes', 0]:.3f}, {model_log.conf_int().loc['children_yes', 1]:.3f}]")
print(f"  p-value: {model_log.pvalues['children_yes']:.4f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY OF FINDINGS")
print("-" * 80)

print(f"""
Sample: {len(df)} married individuals (1969 Psychology Today survey)

Key finding:
- Raw difference: Those without children report {diff_means:.3f} more affair occasions on average
- After controlling for age, years married, gender, religiousness, education,
  occupation, and marriage rating:
  * Children coefficient: {model_b.params['children_yes']:.3f} (95% CI: [{model_b.conf_int().loc['children_yes', 0]:.3f}, {model_b.conf_int().loc['children_yes', 1]:.3f}])
  * p-value: {model_b.pvalues['children_yes']:.4f}
  * Direction: {"Negative (protective)" if model_b.params['children_yes'] < 0 else "Positive (risk factor)"}
  * Statistical significance: {"Yes" if model_b.pvalues['children_yes'] < 0.05 else "No"} (at α=0.05)

Interpretation: Having children is {"associated with" if model_b.pvalues['children_yes'] < 0.05 else "not significantly associated with"}
{"fewer" if model_b.params['children_yes'] < 0 else "more"} extramarital affairs, after adjusting for confounders.

Note: This is an observational study - associations do not imply causation.
""")

print("=" * 80)
print("Analysis complete. Results saved for report.")
print("=" * 80)
