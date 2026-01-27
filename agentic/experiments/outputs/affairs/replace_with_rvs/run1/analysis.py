import csv
import math
from collections import Counter

def read_csv(filename):
    """Read CSV file and return headers and data rows"""
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    return data

def mean(values):
    """Calculate mean of a list of numbers"""
    return sum(values) / len(values)

def median(values):
    """Calculate median of a list of numbers"""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 0:
        return (sorted_vals[n//2-1] + sorted_vals[n//2]) / 2
    else:
        return sorted_vals[n//2]

def std(values):
    """Calculate standard deviation"""
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)

def mann_whitney_u(x, y):
    """Simple Mann-Whitney U test implementation"""
    nx, ny = len(x), len(y)
    # Combine and rank
    combined = [(val, 0) for val in x] + [(val, 1) for val in y]
    combined.sort(key=lambda x: x[0])

    # Assign ranks (handling ties by averaging ranks)
    ranks = []
    i = 0
    while i < len(combined):
        j = i
        # Find all values equal to current value
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        # Assign average rank to all tied values
        avg_rank = (i + j + 1) / 2  # +1 because ranks start at 1
        for k in range(i, j):
            ranks.append((avg_rank, combined[k][1]))
        i = j

    # Sum ranks for group 0 (x)
    rank_sum_x = sum(r[0] for r in ranks if r[1] == 0)

    # Calculate U statistic
    u_x = rank_sum_x - (nx * (nx + 1)) / 2
    u_y = nx * ny - u_x
    u = min(u_x, u_y)

    # Approximate p-value using normal approximation
    mean_u = nx * ny / 2
    std_u = math.sqrt(nx * ny * (nx + ny + 1) / 12)
    z = (u - mean_u) / std_u

    # Two-tailed p-value (approximate)
    p_value = 2 * (1 - normal_cdf(abs(z)))

    return u, p_value

def normal_cdf(x):
    """Cumulative distribution function for standard normal distribution"""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def t_test(x, y):
    """Independent samples t-test"""
    n1, n2 = len(x), len(y)
    mean1, mean2 = mean(x), mean(y)
    std1, std2 = std(x), std(y)

    # Pooled standard deviation
    pooled_std = math.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2))

    # t-statistic
    t = (mean1 - mean2) / (pooled_std * math.sqrt(1/n1 + 1/n2))

    # Degrees of freedom
    df = n1 + n2 - 2

    # Approximate p-value (simplified, assumes large sample)
    z = abs(t)
    p_value = 2 * (1 - normal_cdf(z))

    return t, p_value

def chi_square_test(contingency_table):
    """Chi-square test for independence"""
    # contingency_table is [[a, b], [c, d]]
    a, b = contingency_table[0]
    c, d = contingency_table[1]
    n = a + b + c + d

    # Expected frequencies
    e_a = (a + b) * (a + c) / n
    e_b = (a + b) * (b + d) / n
    e_c = (c + d) * (a + c) / n
    e_d = (c + d) * (b + d) / n

    # Chi-square statistic
    chi2 = ((a - e_a)**2 / e_a + (b - e_b)**2 / e_b +
            (c - e_c)**2 / e_c + (d - e_d)**2 / e_d)

    # Approximate p-value for df=1 using chi-square distribution approximation
    # For df=1, we can use the relationship with normal distribution
    z = math.sqrt(chi2)
    p_value = 2 * (1 - normal_cdf(z))

    return chi2, p_value

# Load the data
print("Loading data...")
data = read_csv('affairs.csv')

print(f"Dataset shape: {len(data)} rows, {len(data[0])} columns")
print(f"\nColumns: {', '.join(data[0].keys())}")

# Extract relevant columns
affairs_with_children = []
affairs_without_children = []

for row in data:
    affairs_val = float(row['affairs'])
    children_val = row['children']

    if children_val == 'yes':
        affairs_with_children.append(affairs_val)
    elif children_val == 'no':
        affairs_without_children.append(affairs_val)

print("\n" + "="*60)
print("ANALYSIS: Does having children affect engagement in affairs?")
print("="*60)

print("\n--- Sample Sizes ---")
print(f"Participants with children: {len(affairs_with_children)}")
print(f"Participants without children: {len(affairs_without_children)}")

print("\n--- Descriptive Statistics ---")
print("\nAffairs engagement - WITH children:")
print(f"  Mean: {mean(affairs_with_children):.3f}")
print(f"  Median: {median(affairs_with_children):.3f}")
print(f"  Std: {std(affairs_with_children):.3f}")
print(f"  Min: {min(affairs_with_children):.1f}")
print(f"  Max: {max(affairs_with_children):.1f}")

print("\nAffairs engagement - WITHOUT children:")
print(f"  Mean: {mean(affairs_without_children):.3f}")
print(f"  Median: {median(affairs_without_children):.3f}")
print(f"  Std: {std(affairs_without_children):.3f}")
print(f"  Min: {min(affairs_without_children):.1f}")
print(f"  Max: {max(affairs_without_children):.1f}")

# Calculate the difference
mean_diff = mean(affairs_without_children) - mean(affairs_with_children)
print(f"\nDifference in means (no children - has children): {mean_diff:.3f}")

# Count people with any affairs
affairs_count_with = sum(1 for x in affairs_with_children if x > 0)
affairs_count_without = sum(1 for x in affairs_without_children if x > 0)

print("\n--- Proportion with any affairs ---")
print(f"With children: {affairs_count_with}/{len(affairs_with_children)} ({100*affairs_count_with/len(affairs_with_children):.1f}%)")
print(f"Without children: {affairs_count_without}/{len(affairs_without_children)} ({100*affairs_count_without/len(affairs_without_children):.1f}%)")

# Statistical tests
print("\n--- Statistical Tests ---")

# Mann-Whitney U test
u_stat, p_value_mw = mann_whitney_u(affairs_with_children, affairs_without_children)
print(f"\nMann-Whitney U test:")
print(f"  U-statistic: {u_stat:.2f}")
print(f"  p-value: {p_value_mw:.4f}")

# Independent t-test
t_stat, p_value_t = t_test(affairs_with_children, affairs_without_children)
print(f"\nIndependent t-test:")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_value_t:.4f}")

# Chi-square test
no_affair_with = len(affairs_with_children) - affairs_count_with
no_affair_without = len(affairs_without_children) - affairs_count_without
contingency = [[affairs_count_with, no_affair_with],
               [affairs_count_without, no_affair_without]]

print("\nContingency table:")
print(f"                  Has Affairs  No Affairs")
print(f"With children:    {affairs_count_with:>11}  {no_affair_with:>10}")
print(f"Without children: {affairs_count_without:>11}  {no_affair_without:>10}")

chi2_stat, p_value_chi = chi_square_test(contingency)
print(f"\nChi-square test:")
print(f"  Chi-square statistic: {chi2_stat:.3f}")
print(f"  p-value: {p_value_chi:.4f}")

# Effect size (Cohen's d)
pooled_std = math.sqrt(((len(affairs_with_children)-1)*std(affairs_with_children)**2 +
                        (len(affairs_without_children)-1)*std(affairs_without_children)**2) /
                       (len(affairs_with_children) + len(affairs_without_children) - 2))
cohens_d = (mean(affairs_with_children) - mean(affairs_without_children)) / pooled_std
print(f"\nCohen's d effect size: {cohens_d:.3f}")
if abs(cohens_d) < 0.2:
    effect_interpretation = "negligible"
elif abs(cohens_d) < 0.5:
    effect_interpretation = "small"
elif abs(cohens_d) < 0.8:
    effect_interpretation = "medium"
else:
    effect_interpretation = "large"
print(f"  Interpretation: {effect_interpretation} effect")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Mean affairs frequency:")
print(f"  With children: {mean(affairs_with_children):.3f}")
print(f"  Without children: {mean(affairs_without_children):.3f}")
print(f"  Difference: {mean_diff:.3f}")
print(f"\nStatistical significance (Mann-Whitney p-value): {p_value_mw:.4f}")
if p_value_mw < 0.05:
    print("  Result: Statistically significant difference (p < 0.05)")
else:
    print("  Result: NOT statistically significant (p >= 0.05)")
print(f"\nEffect size: {cohens_d:.3f} ({effect_interpretation})")

# Interpretation
print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
if mean_diff > 0:
    direction = "People WITHOUT children have HIGHER engagement in affairs"
    answer = "YES, having children decreases engagement in affairs"
elif mean_diff < 0:
    direction = "People WITH children have HIGHER engagement in affairs"
    answer = "NO, having children does NOT decrease engagement in affairs"
else:
    direction = "No difference between groups"
    answer = "NO clear difference"

print(f"\nDirection of effect: {direction}")
print(f"\nAnswer to research question: {answer}")

if p_value_mw < 0.05:
    print(f"This difference is statistically significant (p = {p_value_mw:.4f})")
else:
    print(f"This difference is NOT statistically significant (p = {p_value_mw:.4f})")
    print("We cannot confidently conclude there is a real effect.")
