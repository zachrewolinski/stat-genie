import csv
import math
from collections import defaultdict

# Helper functions for statistical calculations
def mean(values):
    return sum(values) / len(values) if values else 0

def median(values):
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 0:
        return 0
    if n % 2 == 0:
        return (sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2
    return sorted_vals[n//2]

def std(values):
    if len(values) < 2:
        return 0
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)

def mann_whitney_u(group1, group2):
    """Simple Mann-Whitney U test implementation"""
    n1, n2 = len(group1), len(group2)

    # Combine and rank
    combined = [(val, 1) for val in group1] + [(val, 2) for val in group2]
    combined.sort(key=lambda x: x[0])

    # Assign ranks (handling ties with average ranks)
    ranks = []
    i = 0
    while i < len(combined):
        j = i
        # Find all values equal to current
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        # Assign average rank to all tied values
        avg_rank = (i + j + 1) / 2  # +1 because ranks start at 1
        for k in range(i, j):
            ranks.append((avg_rank, combined[k][1]))
        i = j

    # Sum ranks for group 1
    R1 = sum(rank for rank, group in ranks if group == 1)

    # Calculate U statistics
    U1 = R1 - n1 * (n1 + 1) / 2
    U2 = n1 * n2 - U1
    U = min(U1, U2)

    # Calculate z-score for large samples
    mean_U = n1 * n2 / 2
    std_U = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (U - mean_U) / std_U if std_U > 0 else 0

    # Approximate p-value (two-tailed)
    # Using normal approximation
    p_value = 2 * (1 - norm_cdf(abs(z)))

    return U, p_value

def norm_cdf(x):
    """Cumulative distribution function for standard normal distribution"""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def t_test(group1, group2):
    """Independent samples t-test"""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = mean(group1), mean(group2)
    std1, std2 = std(group1), std(group2)

    # Pooled standard deviation
    pooled_var = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
    pooled_std = math.sqrt(pooled_var)

    # t-statistic
    t = (mean1 - mean2) / (pooled_std * math.sqrt(1/n1 + 1/n2)) if pooled_std > 0 else 0

    # Degrees of freedom
    df = n1 + n2 - 2

    # Approximate p-value (two-tailed) using normal approximation for large df
    z = abs(t)
    p_value = 2 * (1 - norm_cdf(z))

    return t, p_value

def chi_square_test(contingency_table):
    """Chi-square test of independence"""
    # contingency_table: [[a, b], [c, d]]
    a, b = contingency_table[0]
    c, d = contingency_table[1]

    n = a + b + c + d

    # Expected frequencies
    row1_sum = a + b
    row2_sum = c + d
    col1_sum = a + c
    col2_sum = b + d

    expected = [
        [row1_sum * col1_sum / n, row1_sum * col2_sum / n],
        [row2_sum * col1_sum / n, row2_sum * col2_sum / n]
    ]

    # Chi-square statistic
    chi2 = 0
    observed = [[a, b], [c, d]]
    for i in range(2):
        for j in range(2):
            if expected[i][j] > 0:
                chi2 += (observed[i][j] - expected[i][j])**2 / expected[i][j]

    # Degrees of freedom
    df = 1

    # Approximate p-value using chi-square distribution
    # For df=1, using Wilson-Hilferty transformation
    z = math.sqrt(chi2)
    p_value = 2 * (1 - norm_cdf(z))

    return chi2, p_value, contingency_table

# Load the data
data = []
with open('affairs.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

print("=" * 80)
print("ANALYSIS: Does having children decrease engagement in extramarital affairs?")
print("=" * 80)
print()

# Basic data exploration
print(f"Dataset size: {len(data)} rows")
print()

# Convert affairs to numeric and group by children
with_children_affairs = []
without_children_affairs = []

for row in data:
    affairs_val = float(row['affairs'])
    if row['children'] == 'yes':
        with_children_affairs.append(affairs_val)
    else:
        without_children_affairs.append(affairs_val)

print("=" * 80)
print("DESCRIPTIVE STATISTICS")
print("=" * 80)
print()

print("WITH children:")
print(f"  Count: {len(with_children_affairs)}")
print(f"  Mean: {mean(with_children_affairs):.3f}")
print(f"  Median: {median(with_children_affairs):.1f}")
print(f"  Std Dev: {std(with_children_affairs):.3f}")
print(f"  Min: {min(with_children_affairs):.1f}")
print(f"  Max: {max(with_children_affairs):.1f}")
with_children_had_affairs = sum(1 for x in with_children_affairs if x > 0)
print(f"  Had affairs (>0): {with_children_had_affairs} ({100*with_children_had_affairs/len(with_children_affairs):.1f}%)")
print()

print("WITHOUT children:")
print(f"  Count: {len(without_children_affairs)}")
print(f"  Mean: {mean(without_children_affairs):.3f}")
print(f"  Median: {median(without_children_affairs):.1f}")
print(f"  Std Dev: {std(without_children_affairs):.3f}")
print(f"  Min: {min(without_children_affairs):.1f}")
print(f"  Max: {max(without_children_affairs):.1f}")
without_children_had_affairs = sum(1 for x in without_children_affairs if x > 0)
print(f"  Had affairs (>0): {without_children_had_affairs} ({100*without_children_had_affairs/len(without_children_affairs):.1f}%)")
print()

# Calculate difference
mean_diff = mean(without_children_affairs) - mean(with_children_affairs)
print(f"Difference in means (without - with): {mean_diff:.3f}")
print()

print("=" * 80)
print("STATISTICAL TESTS")
print("=" * 80)
print()

# Mann-Whitney U test
print("1. Mann-Whitney U Test (non-parametric)")
print("   Null hypothesis: The distributions are the same")
u_stat, p_value_mw = mann_whitney_u(without_children_affairs, with_children_affairs)
print(f"   U-statistic: {u_stat:.2f}")
print(f"   p-value: {p_value_mw:.4f}")
if p_value_mw < 0.05:
    print("   Result: Statistically significant difference (p < 0.05)")
else:
    print("   Result: No statistically significant difference (p >= 0.05)")
print()

# Independent samples t-test
print("2. Independent Samples t-test")
print("   Null hypothesis: Mean affairs are equal between groups")
t_stat, p_value_t = t_test(without_children_affairs, with_children_affairs)
print(f"   t-statistic: {t_stat:.3f}")
print(f"   p-value: {p_value_t:.4f}")
if p_value_t < 0.05:
    print("   Result: Statistically significant difference (p < 0.05)")
else:
    print("   Result: No statistically significant difference (p >= 0.05)")
print()

# Chi-square test
print("3. Chi-Square Test (for proportion with any affairs)")
print("   Null hypothesis: Proportion with affairs is independent of having children")

# Build contingency table
# Row 1: with children, Row 2: without children
# Col 1: no affairs (0), Col 2: had affairs (>0)
with_no_affairs = sum(1 for x in with_children_affairs if x == 0)
with_had_affairs = sum(1 for x in with_children_affairs if x > 0)
without_no_affairs = sum(1 for x in without_children_affairs if x == 0)
without_had_affairs = sum(1 for x in without_children_affairs if x > 0)

contingency = [[with_no_affairs, with_had_affairs],
               [without_no_affairs, without_had_affairs]]

print("\n   Contingency table:")
print(f"                  No Affairs  Had Affairs")
print(f"   With children:     {with_no_affairs:3d}         {with_had_affairs:3d}")
print(f"   Without children:  {without_no_affairs:3d}         {without_had_affairs:3d}")

chi2_stat, p_value_chi, cont_table = chi_square_test(contingency)
print(f"\n   Chi-square statistic: {chi2_stat:.3f}")
print(f"   p-value: {p_value_chi:.4f}")
if p_value_chi < 0.05:
    print("   Result: Statistically significant association (p < 0.05)")
else:
    print("   Result: No statistically significant association (p >= 0.05)")
print()

# Cohen's d effect size
pooled_std = math.sqrt(((len(with_children_affairs) - 1) * std(with_children_affairs)**2 +
                         (len(without_children_affairs) - 1) * std(without_children_affairs)**2) /
                        (len(with_children_affairs) + len(without_children_affairs) - 2))
cohens_d = (mean(without_children_affairs) - mean(with_children_affairs)) / pooled_std if pooled_std > 0 else 0

print("=" * 80)
print("EFFECT SIZE")
print("=" * 80)
print(f"Cohen's d: {cohens_d:.3f}")
if abs(cohens_d) < 0.2:
    effect_interpretation = "negligible"
elif abs(cohens_d) < 0.5:
    effect_interpretation = "small"
elif abs(cohens_d) < 0.8:
    effect_interpretation = "medium"
else:
    effect_interpretation = "large"
print(f"Interpretation: {effect_interpretation} effect size")
print()

print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()

mean_with = mean(with_children_affairs)
mean_without = mean(without_children_affairs)

if mean_with < mean_without:
    direction = "LOWER"
    answer = "YES"
    print(f"People WITH children have {direction} mean affair rates ({mean_with:.3f})")
    print(f"compared to those WITHOUT children ({mean_without:.3f}).")
else:
    direction = "HIGHER or EQUAL"
    answer = "NO"
    print(f"People WITH children have {direction} mean affair rates ({mean_with:.3f})")
    print(f"compared to those WITHOUT children ({mean_without:.3f}).")
print()

if p_value_mw < 0.05 or p_value_chi < 0.05:
    print(f"The difference is STATISTICALLY SIGNIFICANT.")
    print(f"\nAnswer to research question: {answer}")
else:
    print(f"The difference is NOT statistically significant (p >= 0.05).")
    print(f"\nAnswer to research question: Insufficient evidence - {answer}")
print()

print("=" * 80)
