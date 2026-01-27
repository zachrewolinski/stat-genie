import csv
import math
from collections import defaultdict

# Load the dataset
data = []
with open('affairs.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

# Research Question: Does having children decrease (if at all) the engagement in extramarital affairs?

print("="*80)
print("Analysis: Effect of Having Children on Extramarital Affairs")
print("="*80)
print()

# Basic dataset info
print(f"Total observations: {len(data)}")
print()

# Count children distribution
children_counts = defaultdict(int)
for row in data:
    children_counts[row['children']] += 1

print("Children distribution:")
for key, count in children_counts.items():
    print(f"  {key}: {count}")
print()

# Split data by children status
with_children = [row for row in data if row['children'] == 'yes']
without_children = [row for row in data if row['children'] == 'no']

print(f"Observations with children: {len(with_children)}")
print(f"Observations without children: {len(without_children)}")
print()

# Extract affairs values
affairs_with = [float(row['affairs']) for row in with_children]
affairs_without = [float(row['affairs']) for row in without_children]

# Calculate statistics
def mean(values):
    return sum(values) / len(values)

def median(values):
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 0:
        return (sorted_vals[n//2-1] + sorted_vals[n//2]) / 2
    else:
        return sorted_vals[n//2]

def std_dev(values):
    m = mean(values)
    variance = sum((x - m)**2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)

# Calculate affair statistics for each group
print("="*80)
print("Affairs Statistics by Children Status")
print("="*80)
print()

# Mean affairs
mean_affairs_with = mean(affairs_with)
mean_affairs_without = mean(affairs_without)

print(f"Mean affairs (with children): {mean_affairs_with:.4f}")
print(f"Mean affairs (without children): {mean_affairs_without:.4f}")
print(f"Difference: {mean_affairs_without - mean_affairs_with:.4f}")
print()

# Median affairs
median_affairs_with = median(affairs_with)
median_affairs_without = median(affairs_without)

print(f"Median affairs (with children): {median_affairs_with:.4f}")
print(f"Median affairs (without children): {median_affairs_without:.4f}")
print()

# Standard deviation
std_affairs_with = std_dev(affairs_with)
std_affairs_without = std_dev(affairs_without)

print(f"Std Dev affairs (with children): {std_affairs_with:.4f}")
print(f"Std Dev affairs (without children): {std_affairs_without:.4f}")
print()

# Percentage with no affairs
no_affairs_with = sum(1 for x in affairs_with if x == 0)
no_affairs_without = sum(1 for x in affairs_without if x == 0)
pct_no_affairs_with = no_affairs_with / len(affairs_with) * 100
pct_no_affairs_without = no_affairs_without / len(affairs_without) * 100

print(f"% with NO affairs (with children): {pct_no_affairs_with:.2f}%")
print(f"% with NO affairs (without children): {pct_no_affairs_without:.2f}%")
print()

# Percentage with any affairs
pct_with_affairs_with = 100 - pct_no_affairs_with
pct_with_affairs_without = 100 - pct_no_affairs_without

print(f"% with ANY affairs (with children): {pct_with_affairs_with:.2f}%")
print(f"% with ANY affairs (without children): {pct_with_affairs_without:.2f}%")
print()

# Statistical tests - t-test
print("="*80)
print("Statistical Tests")
print("="*80)
print()

# Independent samples t-test
def t_test(sample1, sample2):
    n1 = len(sample1)
    n2 = len(sample2)
    m1 = mean(sample1)
    m2 = mean(sample2)
    s1 = std_dev(sample1)
    s2 = std_dev(sample2)

    # Pooled standard error
    pooled_se = math.sqrt(s1**2/n1 + s2**2/n2)

    # t-statistic
    t_stat = (m1 - m2) / pooled_se

    # Degrees of freedom (Welch-Satterthwaite approximation)
    df = ((s1**2/n1 + s2**2/n2)**2) / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))

    return t_stat, df

t_stat, df = t_test(affairs_with, affairs_without)

print("Independent Samples T-Test:")
print(f"  Test Statistic: {t_stat:.4f}")
print(f"  Degrees of freedom: {df:.2f}")
print()

# For p-value approximation: |t| > 1.96 is roughly p < 0.05 for large samples
is_significant = abs(t_stat) > 1.96
print(f"  |t| > 1.96? {is_significant} (approximate significance at α=0.05)")
print()

# Effect size: Cohen's d
n1 = len(affairs_with)
n2 = len(affairs_without)
pooled_std = math.sqrt(((n1-1)*std_affairs_with**2 + (n2-1)*std_affairs_without**2) / (n1 + n2 - 2))
cohens_d = (mean_affairs_without - mean_affairs_with) / pooled_std

print(f"Effect Size (Cohen's d): {cohens_d:.4f}")
print(f"  Interpretation: ", end="")
if abs(cohens_d) < 0.2:
    print("negligible/very small")
elif abs(cohens_d) < 0.5:
    print("small")
elif abs(cohens_d) < 0.8:
    print("medium")
else:
    print("large")
print()

# Contingency table for chi-square test
print("Contingency Table (Children vs Any Affairs):")
yes_children_no_affairs = sum(1 for x in affairs_with if x == 0)
yes_children_has_affairs = sum(1 for x in affairs_with if x > 0)
no_children_no_affairs = sum(1 for x in affairs_without if x == 0)
no_children_has_affairs = sum(1 for x in affairs_without if x > 0)

print(f"  With children & No affairs: {yes_children_no_affairs}")
print(f"  With children & Has affairs: {yes_children_has_affairs}")
print(f"  No children & No affairs: {no_children_no_affairs}")
print(f"  No children & Has affairs: {no_children_has_affairs}")
print()

# Chi-square test of independence
total = len(data)
total_with_children = len(with_children)
total_without_children = len(without_children)
total_no_affairs = yes_children_no_affairs + no_children_no_affairs
total_has_affairs = yes_children_has_affairs + no_children_has_affairs

# Expected frequencies
exp_yes_no = (total_with_children * total_no_affairs) / total
exp_yes_has = (total_with_children * total_has_affairs) / total
exp_no_no = (total_without_children * total_no_affairs) / total
exp_no_has = (total_without_children * total_has_affairs) / total

# Chi-square statistic
chi2 = ((yes_children_no_affairs - exp_yes_no)**2 / exp_yes_no +
        (yes_children_has_affairs - exp_yes_has)**2 / exp_yes_has +
        (no_children_no_affairs - exp_no_no)**2 / exp_no_no +
        (no_children_has_affairs - exp_no_has)**2 / exp_no_has)

print("Chi-Square Test of Independence:")
print(f"  Chi-square statistic: {chi2:.4f}")
print(f"  Degrees of freedom: 1")
# For df=1, critical value at α=0.05 is 3.841
chi2_significant = chi2 > 3.841
print(f"  χ² > 3.841? {chi2_significant} (significant at α=0.05)")
print()

# Summary
print("="*80)
print("SUMMARY")
print("="*80)
print()
print(f"Mean affairs - WITH children: {mean_affairs_with:.4f}")
print(f"Mean affairs - WITHOUT children: {mean_affairs_without:.4f}")
print(f"Difference (without - with): {mean_affairs_without - mean_affairs_with:.4f}")
print()

if mean_affairs_with < mean_affairs_without:
    print("✓ People WITH children have FEWER affairs on average")
    direction = "decrease"
    decrease_amount = mean_affairs_without - mean_affairs_with
    pct_decrease = (decrease_amount / mean_affairs_without) * 100
    print(f"  Decrease of {decrease_amount:.4f} affairs ({pct_decrease:.2f}% reduction)")
elif mean_affairs_with > mean_affairs_without:
    print("✗ People WITH children have MORE affairs on average")
    direction = "increase"
else:
    print("= No difference in mean affairs")
    direction = "no change"

print()
print(f"Statistical significance (t-test): |t| = {abs(t_stat):.4f}")
print(f"Is this difference statistically significant? {is_significant}")
print(f"Effect size (Cohen's d): {cohens_d:.4f} ({['negligible/very small', 'small', 'medium', 'large'][min(3, int(abs(cohens_d)/0.2))]})")
print()

if is_significant and direction == "decrease":
    print("CONCLUSION: Having children DOES significantly DECREASE engagement in extramarital affairs.")
elif is_significant and direction == "increase":
    print("CONCLUSION: Having children significantly INCREASES engagement in extramarital affairs.")
else:
    print("CONCLUSION: No statistically significant effect of having children on extramarital affairs.")
print()
