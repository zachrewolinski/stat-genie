import csv
import math
from collections import defaultdict

# Load the dataset
data = []
with open('affairs.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

print("=" * 60)
print("ANALYSIS: Impact of Children on Extramarital Affairs")
print("=" * 60)
print()

# Basic dataset info
print(f"Total observations: {len(data)}")
with_children_count = sum(1 for row in data if row['feature6'] == 'yes')
without_children_count = sum(1 for row in data if row['feature6'] == 'no')
print(f"Observations with children: {with_children_count}")
print(f"Observations without children: {without_children_count}")
print()

# Split data by presence of children
with_children = [float(row['feature2']) for row in data if row['feature6'] == 'yes']
without_children = [float(row['feature2']) for row in data if row['feature6'] == 'no']

# Calculate statistics
def mean(values):
    return sum(values) / len(values)

def median(values):
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 0:
        return (sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2
    else:
        return sorted_vals[n//2]

def std_dev(values):
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)

# Analyze affair rates (feature2)
print("=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)
print()

print("Affair frequency (feature2) statistics:")
print()

# With children stats
mean_with = mean(with_children)
median_with = median(with_children)
std_with = std_dev(with_children)
had_affair_with = sum(1 for x in with_children if x > 0)
no_affair_with = sum(1 for x in with_children if x == 0)

print("WITH CHILDREN:")
print(f"  Mean: {mean_with:.4f}")
print(f"  Median: {median_with:.4f}")
print(f"  Std Dev: {std_with:.4f}")
print(f"  Had affairs (>0): {had_affair_with} ({had_affair_with / len(with_children) * 100:.2f}%)")
print(f"  No affairs (=0): {no_affair_with} ({no_affair_with / len(with_children) * 100:.2f}%)")
print()

# Without children stats
mean_without = mean(without_children)
median_without = median(without_children)
std_without = std_dev(without_children)
had_affair_without = sum(1 for x in without_children if x > 0)
no_affair_without = sum(1 for x in without_children if x == 0)

print("WITHOUT CHILDREN:")
print(f"  Mean: {mean_without:.4f}")
print(f"  Median: {median_without:.4f}")
print(f"  Std Dev: {std_without:.4f}")
print(f"  Had affairs (>0): {had_affair_without} ({had_affair_without / len(without_children) * 100:.2f}%)")
print(f"  No affairs (=0): {no_affair_without} ({no_affair_without / len(without_children) * 100:.2f}%)")
print()

# Calculate the difference
mean_diff = mean_with - mean_without
print(f"Difference in mean affair frequency (with - without children): {mean_diff:.4f}")
print()

# Statistical test: Independent samples t-test
print("=" * 60)
print("STATISTICAL TEST")
print("=" * 60)
print()

n1 = len(with_children)
n2 = len(without_children)

# Pooled standard deviation
pooled_var = ((n1 - 1) * std_with**2 + (n2 - 1) * std_without**2) / (n1 + n2 - 2)
pooled_std = math.sqrt(pooled_var)

# T-statistic
t_stat = (mean_with - mean_without) / (pooled_std * math.sqrt(1/n1 + 1/n2))

# Degrees of freedom
df = n1 + n2 - 2

print("Independent Samples T-Test:")
print(f"  T-statistic: {t_stat:.4f}")
print(f"  Degrees of freedom: {df}")
print(f"  Pooled standard deviation: {pooled_std:.4f}")
print()

# Approximate p-value interpretation (for df > 30, use normal approximation)
# For |t| > 1.96, p < 0.05; for |t| > 2.58, p < 0.01
abs_t = abs(t_stat)
if abs_t > 2.58:
    sig_level = "p < 0.01 (highly significant)"
    is_significant = True
elif abs_t > 1.96:
    sig_level = "p < 0.05 (significant)"
    is_significant = True
else:
    sig_level = "p > 0.05 (not significant)"
    is_significant = False

print(f"  Approximate significance: {sig_level}")
print()

# Effect size (Cohen's d)
cohens_d = (mean_with - mean_without) / pooled_std

print("Effect Size (Cohen's d):")
print(f"  Cohen's d: {cohens_d:.4f}")
print(f"  Interpretation: ", end="")
if abs(cohens_d) < 0.2:
    print("Negligible effect")
elif abs(cohens_d) < 0.5:
    print("Small effect")
elif abs(cohens_d) < 0.8:
    print("Medium effect")
else:
    print("Large effect")
print()

# Chi-square test for proportion
print("Chi-Square Test (for proportion having any affair):")
# Contingency table: [with_children_had_affair, with_children_no_affair]
#                     [without_children_had_affair, without_children_no_affair]
total = n1 + n2
row1_total = n1
row2_total = n2
col1_total = had_affair_with + had_affair_without
col2_total = no_affair_with + no_affair_without

# Expected frequencies
exp_11 = (row1_total * col1_total) / total
exp_12 = (row1_total * col2_total) / total
exp_21 = (row2_total * col1_total) / total
exp_22 = (row2_total * col2_total) / total

# Chi-square statistic
chi2 = (((had_affair_with - exp_11)**2 / exp_11) +
        ((no_affair_with - exp_12)**2 / exp_12) +
        ((had_affair_without - exp_21)**2 / exp_21) +
        ((no_affair_without - exp_22)**2 / exp_22))

print(f"  Chi-square statistic: {chi2:.4f}")
print(f"  Degrees of freedom: 1")

# For df=1, critical value at 0.05 is 3.841
if chi2 > 3.841:
    print(f"  Result: Statistically significant at α=0.05 (chi2 > 3.841)")
else:
    print(f"  Result: Not statistically significant at α=0.05 (chi2 <= 3.841)")
print()

# Summary
print("=" * 60)
print("CONCLUSION")
print("=" * 60)
print()

if mean_diff < 0:
    direction = "LOWER"
    interpretation = "Having children is associated with DECREASED extramarital affair engagement"
else:
    direction = "HIGHER"
    interpretation = "Having children is associated with INCREASED extramarital affair engagement"

print(f"People with children have {direction} average affair frequency ({mean_with:.4f})")
print(f"compared to those without children ({mean_without:.4f}).")
print()
print(interpretation)
print()

if is_significant and mean_diff < 0:
    print("ANSWER: YES - Having children significantly decreases engagement in extramarital affairs.")
    final_answer = "Yes"
elif mean_diff < 0:
    print("ANSWER: The data suggests children may decrease affairs, but not statistically significant.")
    final_answer = "No (not statistically significant)"
else:
    print("ANSWER: NO - Having children does not decrease extramarital affairs.")
    final_answer = "No"

print()
print(f"Final answer for conclusion.txt: {final_answer}")
