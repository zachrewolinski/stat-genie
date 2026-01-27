import csv
import math
from collections import Counter

# Load the dataset
def load_data(filename):
    data = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def mean(values):
    return sum(values) / len(values) if values else 0

def median(values):
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 0:
        return 0
    if n % 2 == 0:
        return (sorted_vals[n//2-1] + sorted_vals[n//2]) / 2
    return sorted_vals[n//2]

def std_dev(values):
    if not values:
        return 0
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return math.sqrt(variance)

def mannwhitneyu(group1, group2):
    """Simple Mann-Whitney U test implementation"""
    n1, n2 = len(group1), len(group2)

    # Combine and rank
    combined = [(val, 1) for val in group1] + [(val, 2) for val in group2]
    combined.sort(key=lambda x: x[0])

    # Assign ranks (handling ties by average rank)
    ranks = []
    i = 0
    while i < len(combined):
        j = i
        # Find all values equal to current
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        # Assign average rank
        avg_rank = (i + j + 1) / 2  # +1 because ranks start at 1
        for k in range(i, j):
            ranks.append((avg_rank, combined[k][1]))
        i = j

    # Sum ranks for group 1
    R1 = sum(rank for rank, group in ranks if group == 1)

    # Calculate U statistic
    U1 = R1 - n1 * (n1 + 1) / 2
    U2 = n1 * n2 - U1
    U = min(U1, U2)

    # Calculate z-score for large samples
    mean_U = n1 * n2 / 2
    std_U = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (U - mean_U) / std_U if std_U > 0 else 0

    # Approximate p-value using normal distribution
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

    return U, p_value

def chi_square_test(observed):
    """Chi-square test for 2x2 contingency table"""
    # observed is [[a, b], [c, d]]
    a, b = observed[0]
    c, d = observed[1]
    n = a + b + c + d

    # Expected frequencies
    exp_a = (a + b) * (a + c) / n
    exp_b = (a + b) * (b + d) / n
    exp_c = (c + d) * (a + c) / n
    exp_d = (c + d) * (b + d) / n

    # Chi-square statistic
    chi2 = ((a - exp_a)**2 / exp_a +
            (b - exp_b)**2 / exp_b +
            (c - exp_c)**2 / exp_c +
            (d - exp_d)**2 / exp_d)

    # Approximate p-value (df=1)
    # Using chi-square CDF approximation
    if chi2 > 10:
        p_value = 0.001
    elif chi2 > 6.635:
        p_value = 0.01
    elif chi2 > 3.841:
        p_value = 0.05
    elif chi2 > 2.706:
        p_value = 0.10
    else:
        p_value = 0.20

    return chi2, p_value

df = load_data('affairs.csv')

print("="*70)
print("ANALYSIS: Effect of Having Children on Extramarital Affairs")
print("="*70)

# Overview of the data
print("\nDataset Overview:")
print(f"Total observations: {len(df)}")

children_count = Counter(row['children'] for row in df)
print(f"\nChildren distribution:")
for status, count in children_count.items():
    print(f"  {status}: {count}")

# Separate data by children status
with_children = [row for row in df if row['children'] == 'yes']
without_children = [row for row in df if row['children'] == 'no']

print(f"\nPeople with children: {len(with_children)}")
print(f"People without children: {len(without_children)}")

# Extract affairs values
affairs_with = [float(row['affairs']) for row in with_children]
affairs_without = [float(row['affairs']) for row in without_children]

# Analyze affairs variable by children status
print("\n" + "="*70)
print("DESCRIPTIVE STATISTICS")
print("="*70)

print("\nAffairs statistics for people WITH children:")
mean_with = mean(affairs_with)
median_with = median(affairs_with)
std_with = std_dev(affairs_with)
prop_with_affair = sum(1 for x in affairs_with if x > 0) / len(affairs_with)
print(f"  Mean: {mean_with:.3f}")
print(f"  Median: {median_with:.3f}")
print(f"  Std Dev: {std_with:.3f}")
print(f"  Proportion with any affair (affairs > 0): {prop_with_affair:.3f}")

print("\nAffairs statistics for people WITHOUT children:")
mean_without = mean(affairs_without)
median_without = median(affairs_without)
std_without = std_dev(affairs_without)
prop_without_affair = sum(1 for x in affairs_without if x > 0) / len(affairs_without)
print(f"  Mean: {mean_without:.3f}")
print(f"  Median: {median_without:.3f}")
print(f"  Std Dev: {std_without:.3f}")
print(f"  Proportion with any affair (affairs > 0): {prop_without_affair:.3f}")

# Calculate difference
mean_diff = mean_without - mean_with
prop_diff = prop_without_affair - prop_with_affair

print("\n" + "="*70)
print("DIFFERENCE ANALYSIS")
print("="*70)
print(f"\nDifference in mean affairs (without - with children): {mean_diff:.3f}")
print(f"Difference in proportion with affairs (without - with children): {prop_diff:.3f}")

# Statistical testing
print("\n" + "="*70)
print("STATISTICAL TESTS")
print("="*70)

# Mann-Whitney U test
U, p_value_mw = mannwhitneyu(affairs_without, affairs_with)
print("\nMann-Whitney U Test (comparing affairs distributions):")
print(f"  U-statistic: {U:.2f}")
print(f"  P-value: {p_value_mw:.4f}")
print(f"  Significant at α=0.05: {'Yes' if p_value_mw < 0.05 else 'No'}")

# Chi-square test for independence
# Count for contingency table
with_yes_affair = sum(1 for x in affairs_with if x > 0)
with_no_affair = len(affairs_with) - with_yes_affair
without_yes_affair = sum(1 for x in affairs_without if x > 0)
without_no_affair = len(affairs_without) - without_yes_affair

print("\nContingency table (children vs any affair):")
print(f"               No Affair  Has Affair")
print(f"With children:     {with_no_affair:4d}      {with_yes_affair:4d}")
print(f"No children:       {without_no_affair:4d}      {without_yes_affair:4d}")

contingency = [[with_no_affair, with_yes_affair],
               [without_no_affair, without_yes_affair]]
chi2, p_value_chi = chi_square_test(contingency)

print(f"\nChi-square Test:")
print(f"  Chi-square statistic: {chi2:.3f}")
print(f"  P-value: <{p_value_chi:.2f}")
print(f"  Significant at α=0.05: {'Yes' if p_value_chi <= 0.05 else 'No'}")

# Detailed breakdown
print("\n" + "="*70)
print("DETAILED BREAKDOWN")
print("="*70)

print("\nDistribution of affairs by children status:")

for child_status, affairs_vals in [('YES', affairs_with), ('NO', affairs_without)]:
    affairs_counter = Counter(affairs_vals)
    print(f"\n{child_status} children:")
    for affairs_val in sorted(affairs_counter.keys()):
        count = affairs_counter[affairs_val]
        pct = count / len(affairs_vals) * 100
        print(f"  Affairs={affairs_val:.0f}: {count} ({pct:.1f}%)")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\nPeople WITHOUT children have:")
print(f"  - {'HIGHER' if mean_diff > 0 else 'LOWER'} mean affair frequency ({abs(mean_diff):.3f} {'more' if mean_diff > 0 else 'less'})")
print(f"  - {'HIGHER' if prop_diff > 0 else 'LOWER'} proportion engaging in affairs ({abs(prop_diff):.3f} or {abs(prop_diff)*100:.1f}% {'more' if prop_diff > 0 else 'less'})")

if p_value_mw < 0.05:
    print(f"\nThe difference IS STATISTICALLY SIGNIFICANT (p < 0.05)")
    if mean_diff < 0:
        print("Conclusion: Having children is associated with INCREASED affair engagement.")
    else:
        print("Conclusion: Having children is associated with DECREASED affair engagement.")
else:
    print(f"\nThe difference is NOT statistically significant (p >= 0.05)")
    print("Conclusion: Cannot conclude that having children affects affair engagement.")
