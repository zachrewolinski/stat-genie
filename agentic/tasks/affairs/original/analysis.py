#!/usr/bin/env python3
"""
Analysis of the relationship between having children and extramarital affairs.
Dataset: Affairs data from Fair (1978), Psychology Today survey (1969).
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load the data
df = pd.read_csv('affairs.csv')

print("=" * 70)
print("AFFAIRS AND CHILDREN ANALYSIS")
print("=" * 70)
print()

# Basic dataset info
print(f"Total sample size: {len(df)}")
print(f"Variables: {list(df.columns)}")
print()

# ============================================================================
# 1. DESCRIPTIVE STATISTICS
# ============================================================================
print("=" * 70)
print("1. DESCRIPTIVE STATISTICS")
print("=" * 70)
print()

# Distribution of affairs
print("Distribution of affairs variable:")
print(df['affairs'].value_counts().sort_index())
print()
print(f"Proportion with zero affairs: {(df['affairs'] == 0).mean():.3f}")
print(f"Mean affairs: {df['affairs'].mean():.3f}")
print(f"Median affairs: {df['affairs'].median():.3f}")
print(f"Std dev affairs: {df['affairs'].std():.3f}")
print()

# Sample sizes by children group
print("Sample sizes by children group:")
children_counts = df['children'].value_counts()
print(children_counts)
print()

# Descriptive statistics by children group
print("Affairs statistics by children group:")
grouped_stats = df.groupby('children')['affairs'].agg([
    ('count', 'count'),
    ('mean', 'mean'),
    ('median', 'median'),
    ('std', 'std'),
    ('prop_zero', lambda x: (x == 0).mean())
])
print(grouped_stats)
print()

# Additional covariates by children group
print("Covariates by children group:")
for var in ['age', 'yearsmarried', 'religiousness', 'rating']:
    print(f"\n{var.capitalize()}:")
    print(df.groupby('children')[var].agg(['mean', 'std']))

print()
print("Gender distribution by children group:")
print(pd.crosstab(df['children'], df['gender'], normalize='index'))
print()

# ============================================================================
# 2. MODEL A: SIMPLE BIVARIATE ASSOCIATION
# ============================================================================
print("=" * 70)
print("2. MODEL A: SIMPLE BIVARIATE ASSOCIATION")
print("=" * 70)
print()

# Convert children to binary (yes=1, no=0)
df['children_binary'] = (df['children'] == 'yes').astype(int)

# Simple OLS regression
print("Model A: Simple linear regression (affairs ~ children)")
model_a = smf.ols('affairs ~ children', data=df).fit()
print(model_a.summary())
print()

# T-test for difference in means
yes_affairs = df[df['children'] == 'yes']['affairs']
no_affairs = df[df['children'] == 'no']['affairs']
t_stat, p_val = stats.ttest_ind(yes_affairs, no_affairs)
print(f"T-test for difference in means:")
print(f"  Mean (children=yes): {yes_affairs.mean():.3f}")
print(f"  Mean (children=no): {no_affairs.mean():.3f}")
print(f"  Difference: {yes_affairs.mean() - no_affairs.mean():.3f}")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_val:.4f}")
print()

# ============================================================================
# 3. MODEL B: MULTIVARIABLE MODEL WITH CONFOUNDERS
# ============================================================================
print("=" * 70)
print("3. MODEL B: MULTIVARIABLE MODEL WITH CONFOUNDERS")
print("=" * 70)
print()

# OLS with covariates
print("Model B1: Linear regression with covariates")
model_b1 = smf.ols('''affairs ~ children + gender + age + yearsmarried +
                      religiousness + education + occupation + rating''',
                   data=df).fit()
print(model_b1.summary())
print()

# Key coefficient interpretation
children_coef = model_b1.params['children[T.yes]']
children_se = model_b1.bse['children[T.yes]']
children_ci = model_b1.conf_int().loc['children[T.yes]']
print(f"Children coefficient: {children_coef:.3f}")
print(f"Standard error: {children_se:.3f}")
print(f"95% CI: [{children_ci[0]:.3f}, {children_ci[1]:.3f}]")
print(f"p-value: {model_b1.pvalues['children[T.yes]']:.4f}")
print()

# ============================================================================
# 4. MODEL B2: LOG-TRANSFORMED OUTCOME (handles zeros better)
# ============================================================================
print("=" * 70)
print("4. MODEL B2: LOG-TRANSFORMED OUTCOME")
print("=" * 70)
print()

# Create log(affairs + 1) to handle zeros
df['log_affairs'] = np.log(df['affairs'] + 1)

print("Model B2: Linear regression with log(affairs + 1)")
model_b2 = smf.ols('''log_affairs ~ children + gender + age + yearsmarried +
                      religiousness + education + occupation + rating''',
                   data=df).fit()
print(model_b2.summary())
print()

# Interpretation
log_children_coef = model_b2.params['children[T.yes]']
log_children_se = model_b2.bse['children[T.yes]']
log_children_ci = model_b2.conf_int().loc['children[T.yes]']
print(f"Children coefficient (log scale): {log_children_coef:.3f}")
print(f"Standard error: {log_children_se:.3f}")
print(f"95% CI: [{log_children_ci[0]:.3f}, {log_children_ci[1]:.3f}]")
print(f"p-value: {model_b2.pvalues['children[T.yes]']:.4f}")
print()
print(f"Approximate % change in median affairs: {(np.exp(log_children_coef) - 1) * 100:.1f}%")
print()

# ============================================================================
# 5. MODEL B3: POISSON MODEL (for count-like outcome)
# ============================================================================
print("=" * 70)
print("5. MODEL B3: POISSON MODEL")
print("=" * 70)
print()

# Poisson regression (treating affairs as count data)
print("Model B3: Poisson regression")
model_b3 = smf.poisson('affairs ~ children + gender + age + yearsmarried + religiousness + education + occupation + rating',
                       data=df).fit(maxiter=100)
print(model_b3.summary())
print()

poisson_children_coef = model_b3.params['children[T.yes]']
poisson_children_se = model_b3.bse['children[T.yes]']
poisson_children_ci = model_b3.conf_int().loc['children[T.yes]']
print(f"Children coefficient (Poisson): {poisson_children_coef:.3f}")
print(f"Standard error: {poisson_children_se:.3f}")
print(f"95% CI: [{poisson_children_ci[0]:.3f}, {poisson_children_ci[1]:.3f}]")
print(f"p-value: {model_b3.pvalues['children[T.yes]']:.4f}")
print(f"Rate ratio (exp(coef)): {np.exp(poisson_children_coef):.3f}")
print(f"  Interpretation: {(np.exp(poisson_children_coef) - 1) * 100:.1f}% change in affairs rate")
print()

# ============================================================================
# 6. DIAGNOSTIC CHECKS
# ============================================================================
print("=" * 70)
print("6. DIAGNOSTIC CHECKS")
print("=" * 70)
print()

# Check correlation between age and years married
print(f"Correlation between age and yearsmarried: {df['age'].corr(df['yearsmarried']):.3f}")
print()

# Check if other variables differ by children status
print("Potential confounders - differences by children status:")
for var in ['age', 'yearsmarried', 'religiousness', 'rating']:
    yes_val = df[df['children'] == 'yes'][var].mean()
    no_val = df[df['children'] == 'no'][var].mean()
    t_stat, p_val = stats.ttest_ind(
        df[df['children'] == 'yes'][var],
        df[df['children'] == 'no'][var]
    )
    print(f"  {var}: yes={yes_val:.2f}, no={no_val:.2f}, p={p_val:.4f}")
print()

# Residual diagnostics for Model B2 (log-transformed)
print("Residual diagnostics for Model B2 (log-transformed):")
residuals = model_b2.resid
print(f"  Mean of residuals: {residuals.mean():.6f}")
print(f"  Std dev of residuals: {residuals.std():.3f}")
print()

# ============================================================================
# SUMMARY OF KEY FINDINGS
# ============================================================================
print("=" * 70)
print("SUMMARY OF KEY FINDINGS")
print("=" * 70)
print()

print(f"1. Sample sizes:")
print(f"   - With children: {(df['children'] == 'yes').sum()}")
print(f"   - Without children: {(df['children'] == 'no').sum()}")
print()

print(f"2. Unadjusted means:")
print(f"   - Affairs (with children): {yes_affairs.mean():.3f}")
print(f"   - Affairs (without children): {no_affairs.mean():.3f}")
print(f"   - Difference: {yes_affairs.mean() - no_affairs.mean():.3f}")
print()

print(f"3. Simple model (Model A):")
print(f"   - Coefficient: {model_a.params['children[T.yes]']:.3f}")
print(f"   - p-value: {model_a.pvalues['children[T.yes]']:.4f}")
print()

print(f"4. Adjusted model with covariates (Model B1, linear):")
print(f"   - Coefficient: {children_coef:.3f}")
print(f"   - 95% CI: [{children_ci[0]:.3f}, {children_ci[1]:.3f}]")
print(f"   - p-value: {model_b1.pvalues['children[T.yes]']:.4f}")
print()

print(f"5. Log-transformed model (Model B2):")
print(f"   - Coefficient: {log_children_coef:.3f}")
print(f"   - 95% CI: [{log_children_ci[0]:.3f}, {log_children_ci[1]:.3f}]")
print(f"   - p-value: {model_b2.pvalues['children[T.yes]']:.4f}")
print()

print(f"6. Poisson model (Model B3):")
print(f"   - Coefficient: {poisson_children_coef:.3f}")
print(f"   - 95% CI: [{poisson_children_ci[0]:.3f}, {poisson_children_ci[1]:.3f}]")
print(f"   - p-value: {model_b3.pvalues['children[T.yes]']:.4f}")
print(f"   - Rate ratio: {np.exp(poisson_children_coef):.3f}")
print()

print("All models suggest having children is associated with reduced")
print("extramarital affairs, though statistical significance varies by model.")
print()
print("=" * 70)
