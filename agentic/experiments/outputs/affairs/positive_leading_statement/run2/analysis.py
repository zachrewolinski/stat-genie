import pandas as pd
import numpy as np
from scipy import stats

# Load the dataset
df = pd.read_csv('affairs.csv')

print("="*80)
print("ANALYSIS: Does having children decrease engagement in extramarital affairs?")
print("="*80)

# Basic dataset info
print(f"\nDataset size: {len(df)} observations")
print(f"\nColumns: {list(df.columns)}")

# Group by children status
children_yes = df[df['children'] == 'yes']
children_no = df[df['children'] == 'no']

print(f"\nNumber with children: {len(children_yes)}")
print(f"Number without children: {len(children_no)}")

# Calculate affair statistics for each group
print("\n" + "="*80)
print("AFFAIR ENGAGEMENT STATISTICS")
print("="*80)

# For those with children
affairs_with_children = children_yes['affairs']
mean_affairs_with_children = affairs_with_children.mean()
median_affairs_with_children = affairs_with_children.median()
percent_any_affairs_with_children = (affairs_with_children > 0).sum() / len(affairs_with_children) * 100

print("\nWith Children:")
print(f"  Mean affair score: {mean_affairs_with_children:.4f}")
print(f"  Median affair score: {median_affairs_with_children:.4f}")
print(f"  Percentage with any affairs: {percent_any_affairs_with_children:.2f}%")
print(f"  Number with any affairs: {(affairs_with_children > 0).sum()}")
print(f"  Number with no affairs: {(affairs_with_children == 0).sum()}")

# For those without children
affairs_without_children = children_no['affairs']
mean_affairs_without_children = affairs_without_children.mean()
median_affairs_without_children = affairs_without_children.median()
percent_any_affairs_without_children = (affairs_without_children > 0).sum() / len(affairs_without_children) * 100

print("\nWithout Children:")
print(f"  Mean affair score: {mean_affairs_without_children:.4f}")
print(f"  Median affair score: {median_affairs_without_children:.4f}")
print(f"  Percentage with any affairs: {percent_any_affairs_without_children:.2f}%")
print(f"  Number with any affairs: {(affairs_without_children > 0).sum()}")
print(f"  Number with no affairs: {(affairs_without_children == 0).sum()}")

# Calculate differences
print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"\nDifference in mean (With Children - Without Children): {mean_affairs_with_children - mean_affairs_without_children:.4f}")
print(f"Difference in median (With Children - Without Children): {median_affairs_with_children - median_affairs_without_children:.4f}")
print(f"Difference in percentage with any affairs: {percent_any_affairs_with_children - percent_any_affairs_without_children:.2f} percentage points")

# Statistical tests
print("\n" + "="*80)
print("STATISTICAL TESTS")
print("="*80)

# Mann-Whitney U test (non-parametric, good for non-normal distributions)
statistic, p_value_mann_whitney = stats.mannwhitneyu(affairs_with_children, affairs_without_children, alternative='two-sided')
print(f"\nMann-Whitney U Test:")
print(f"  Test statistic: {statistic:.4f}")
print(f"  P-value: {p_value_mann_whitney:.6f}")
print(f"  Significant at α=0.05? {'Yes' if p_value_mann_whitney < 0.05 else 'No'}")

# One-sided test to check if having children DECREASES affairs
statistic_less, p_value_less = stats.mannwhitneyu(affairs_with_children, affairs_without_children, alternative='less')
print(f"\nMann-Whitney U Test (one-sided: with children < without children):")
print(f"  Test statistic: {statistic_less:.4f}")
print(f"  P-value: {p_value_less:.6f}")
print(f"  Significant at α=0.05? {'Yes' if p_value_less < 0.05 else 'No'}")

# T-test (parametric)
t_statistic, p_value_ttest = stats.ttest_ind(affairs_with_children, affairs_without_children)
print(f"\nIndependent T-Test (two-sided):")
print(f"  Test statistic: {t_statistic:.4f}")
print(f"  P-value: {p_value_ttest:.6f}")
print(f"  Significant at α=0.05? {'Yes' if p_value_ttest < 0.05 else 'No'}")

# Chi-square test for binary outcome (any affairs vs no affairs)
contingency_table = pd.crosstab(df['children'], df['affairs'] > 0)
print("\nContingency Table (Children vs Any Affairs):")
print(contingency_table)

chi2, p_value_chi2, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-Square Test:")
print(f"  Chi-square statistic: {chi2:.4f}")
print(f"  P-value: {p_value_chi2:.6f}")
print(f"  Degrees of freedom: {dof}")
print(f"  Significant at α=0.05? {'Yes' if p_value_chi2 < 0.05 else 'No'}")

# Effect size (Cohen's d)
pooled_std = np.sqrt(((len(affairs_with_children) - 1) * affairs_with_children.std()**2 +
                       (len(affairs_without_children) - 1) * affairs_without_children.std()**2) /
                      (len(affairs_with_children) + len(affairs_without_children) - 2))
cohens_d = (mean_affairs_with_children - mean_affairs_without_children) / pooled_std

print(f"\nEffect Size (Cohen's d): {cohens_d:.4f}")
print(f"  Interpretation: ", end="")
if abs(cohens_d) < 0.2:
    print("Negligible effect")
elif abs(cohens_d) < 0.5:
    print("Small effect")
elif abs(cohens_d) < 0.8:
    print("Medium effect")
else:
    print("Large effect")

# Controlling for confounding variables - correlation analysis
print("\n" + "="*80)
print("CONFOUNDING VARIABLES ANALYSIS")
print("="*80)

# Check correlation between having children and other variables
df['children_numeric'] = (df['children'] == 'yes').astype(int)
df['any_affairs'] = (df['affairs'] > 0).astype(int)

print("\nCorrelation between 'children' and other variables:")
for col in ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']:
    corr = df['children_numeric'].corr(df[col])
    print(f"  {col}: {corr:.4f}")

print("\n" + "="*80)
print("CONCLUSION SUMMARY")
print("="*80)

print(f"\nMean affairs score:")
print(f"  With children: {mean_affairs_with_children:.4f}")
print(f"  Without children: {mean_affairs_without_children:.4f}")
print(f"  Difference: {mean_affairs_with_children - mean_affairs_without_children:.4f}")

print(f"\nPercentage with any affairs:")
print(f"  With children: {percent_any_affairs_with_children:.2f}%")
print(f"  Without children: {percent_any_affairs_without_children:.2f}%")
print(f"  Difference: {percent_any_affairs_with_children - percent_any_affairs_without_children:.2f} percentage points")

if mean_affairs_with_children < mean_affairs_without_children:
    print("\n✓ People with children have LOWER mean affair scores than those without children.")
else:
    print("\n✗ People with children have HIGHER mean affair scores than those without children.")

if percent_any_affairs_with_children < percent_any_affairs_without_children:
    print("✓ People with children have LOWER rates of any affairs than those without children.")
else:
    print("✗ People with children have HIGHER rates of any affairs than those without children.")

print(f"\nStatistical significance (one-sided test): p = {p_value_less:.6f}")
if p_value_less < 0.05:
    print("✓ The difference IS statistically significant at α=0.05")
else:
    print("✗ The difference is NOT statistically significant at α=0.05")

print("\n" + "="*80)
