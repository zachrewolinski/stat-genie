import csv
import math
from collections import defaultdict

# Load the dataset
data = []
with open('affairs.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

print("="*80)
print("RESEARCH QUESTION: Does having children decrease the engagement in extramarital affairs?")
print("="*80)
print()

# Basic dataset info
print("Dataset Overview:")
print(f"Total observations: {len(data)}")
print()

# Separate data by children status
with_children = [row for row in data if row['children'] == 'yes']
without_children = [row for row in data if row['children'] == 'no']

print(f"Observations with children: {len(with_children)}")
print(f"Observations without children: {len(without_children)}")
print()

# Function to calculate mean
def mean(values):
    return sum(values) / len(values) if values else 0

# Function to calculate standard deviation
def std_dev(values):
    if len(values) < 2:
        return 0
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)

# Function to calculate median
def median(values):
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n == 0:
        return 0
    if n % 2 == 0:
        return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
    return sorted_values[n//2]

# Extract affairs values
with_children_affairs = [float(row['affairs']) for row in with_children]
without_children_affairs = [float(row['affairs']) for row in without_children]

# Analyze affairs by children status
print("="*80)
print("DESCRIPTIVE STATISTICS")
print("="*80)
print()

mean_with = mean(with_children_affairs)
median_with = median(with_children_affairs)
std_with = std_dev(with_children_affairs)
any_affair_with = sum(1 for x in with_children_affairs if x > 0) / len(with_children_affairs) * 100

print("Affairs statistics for people WITH children:")
print(f"  Mean affairs: {mean_with:.4f}")
print(f"  Median affairs: {median_with:.4f}")
print(f"  Std dev: {std_with:.4f}")
print(f"  % with any affair (affairs > 0): {any_affair_with:.2f}%")
print()

mean_without = mean(without_children_affairs)
median_without = median(without_children_affairs)
std_without = std_dev(without_children_affairs)
any_affair_without = sum(1 for x in without_children_affairs if x > 0) / len(without_children_affairs) * 100

print("Affairs statistics for people WITHOUT children:")
print(f"  Mean affairs: {mean_without:.4f}")
print(f"  Median affairs: {median_without:.4f}")
print(f"  Std dev: {std_without:.4f}")
print(f"  % with any affair (affairs > 0): {any_affair_without:.2f}%")
print()

# Calculate the difference
mean_diff = mean_with - mean_without
print(f"Difference in mean affairs (with children - without children): {mean_diff:.4f}")
print()

# Perform t-test
print("="*80)
print("STATISTICAL TESTING")
print("="*80)
print()

print("Independent Samples T-Test:")
n1 = len(with_children_affairs)
n2 = len(without_children_affairs)

# Pooled standard deviation
pooled_var = ((n1 - 1) * std_with**2 + (n2 - 1) * std_without**2) / (n1 + n2 - 2)
pooled_std = math.sqrt(pooled_var)

# Standard error
se = pooled_std * math.sqrt(1/n1 + 1/n2)

# T-statistic
t_stat = mean_diff / se if se > 0 else 0

# Degrees of freedom
df = n1 + n2 - 2

print(f"   T-statistic: {t_stat:.4f}")
print(f"   Degrees of freedom: {df}")
print(f"   Standard error: {se:.4f}")

# Approximate p-value assessment (two-tailed)
# For df > 30, we can use normal approximation
# Critical value for α=0.05 (two-tailed) is approximately 1.96
abs_t = abs(t_stat)
if abs_t > 2.576:
    p_approx = "< 0.01 (highly significant)"
    significant = True
elif abs_t > 1.96:
    p_approx = "< 0.05 (significant)"
    significant = True
else:
    p_approx = "> 0.05 (not significant)"
    significant = False

print(f"   P-value (approximate): {p_approx}")
print(f"   Significant at α=0.05? {significant}")
print()

# Chi-square test for categorical analysis
print("Chi-Square Test (categorical approach):")
print("   Tests if having children affects likelihood of ANY affair")

with_children_any = sum(1 for x in with_children_affairs if x > 0)
with_children_none = sum(1 for x in with_children_affairs if x == 0)
without_children_any = sum(1 for x in without_children_affairs if x > 0)
without_children_none = sum(1 for x in without_children_affairs if x == 0)

print(f"                     Any Affair    No Affair")
print(f"   With children:    {with_children_any:>10d}  {with_children_none:>10d}")
print(f"   Without children: {without_children_any:>10d}  {without_children_none:>10d}")

# Calculate chi-square statistic
total = n1 + n2
total_any = with_children_any + without_children_any
total_none = with_children_none + without_children_none

exp_with_any = (n1 * total_any) / total
exp_with_none = (n1 * total_none) / total
exp_without_any = (n2 * total_any) / total
exp_without_none = (n2 * total_none) / total

chi2 = ((with_children_any - exp_with_any)**2 / exp_with_any +
        (with_children_none - exp_with_none)**2 / exp_with_none +
        (without_children_any - exp_without_any)**2 / exp_without_any +
        (without_children_none - exp_without_none)**2 / exp_without_none)

print(f"   Chi-square statistic: {chi2:.4f}")
print(f"   Degrees of freedom: 1")

# Critical value for chi-square with df=1 at α=0.05 is 3.841
chi_significant = chi2 > 3.841
print(f"   Significant at α=0.05? {chi_significant}")
print()

# Effect size (Cohen's d)
print("Effect Size (Cohen's d):")
cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
print(f"   Cohen's d: {cohens_d:.4f}")
print(f"   Interpretation: ", end="")
if abs(cohens_d) < 0.2:
    print("negligible effect")
elif abs(cohens_d) < 0.5:
    print("small effect")
elif abs(cohens_d) < 0.8:
    print("medium effect")
else:
    print("large effect")
print()

# Distribution of affairs by children status
print("="*80)
print("DISTRIBUTION OF AFFAIRS BY CHILDREN STATUS")
print("="*80)
print()

def get_distribution(affairs_list):
    dist = defaultdict(int)
    for val in affairs_list:
        dist[val] += 1
    return dist

with_dist = get_distribution(with_children_affairs)
without_dist = get_distribution(without_children_affairs)

print("People WITH children - Affairs distribution:")
for val in sorted(with_dist.keys()):
    print(f"   {int(val)}: {with_dist[val]}")
print()

print("People WITHOUT children - Affairs distribution:")
for val in sorted(without_dist.keys()):
    print(f"   {int(val)}: {without_dist[val]}")
print()

# Confounding variables analysis
print("="*80)
print("POTENTIAL CONFOUNDING VARIABLES")
print("="*80)
print()

with_years = mean([float(row['yearsmarried']) for row in with_children])
without_years = mean([float(row['yearsmarried']) for row in without_children])

with_age = mean([float(row['age']) for row in with_children])
without_age = mean([float(row['age']) for row in without_children])

with_rating = mean([float(row['rating']) for row in with_children])
without_rating = mean([float(row['rating']) for row in without_children])

print("Years married by children status:")
print(f"  With children - mean years married: {with_years:.2f}")
print(f"  Without children - mean years married: {without_years:.2f}")
print()

print("Age by children status:")
print(f"  With children - mean age: {with_age:.2f}")
print(f"  Without children - mean age: {without_age:.2f}")
print()

print("Marriage rating by children status:")
print(f"  With children - mean rating: {with_rating:.2f}")
print(f"  Without children - mean rating: {without_rating:.2f}")
print()

# Summary and conclusion
print("="*80)
print("SUMMARY OF FINDINGS")
print("="*80)
print()

print(f"1. People WITH children have a mean affairs score of {mean_with:.4f}")
print(f"2. People WITHOUT children have a mean affairs score of {mean_without:.4f}")
print(f"3. Difference: {mean_diff:.4f} (positive means more affairs with children)")
print()

print(f"4. Proportion engaging in ANY affair:")
print(f"   - With children: {any_affair_with:.2f}%")
print(f"   - Without children: {any_affair_without:.2f}%")
print()

print(f"5. Statistical significance:")
print(f"   - T-test p-value (approximate): {p_approx}")
print(f"   - Chi-square test: {'significant' if chi_significant else 'not significant'} at α=0.05")
print()

print(f"6. Effect size (Cohen's d): {cohens_d:.4f}")
print()

print("="*80)
print("INTERPRETATION")
print("="*80)
print()

# Determine the answer based on the analysis
if mean_diff < 0:
    print("Having children is associated with LOWER engagement in extramarital affairs.")
    direction = "decrease"
else:
    print("Having children is associated with HIGHER engagement in extramarital affairs.")
    direction = "increase"

# Check if the effect is statistically significant
if significant or chi_significant:
    print("This difference is STATISTICALLY SIGNIFICANT.")
    if direction == "decrease":
        answer = "Yes"
        reason = "The data shows significantly lower affair rates among those with children."
    else:
        answer = "No"
        reason = "The data shows significantly higher affair rates among those with children."
else:
    print("However, this difference is NOT statistically significant.")
    answer = "No"
    reason = "There is no statistically significant difference in affair rates between those with and without children."

print()
print(f"FINAL ANSWER: {answer}")
print(f"REASONING: {reason}")
print()

# Save the final answer for conclusion file
with open('_analysis_result.txt', 'w') as f:
    f.write(f"{answer}\n")
    f.write(f"{reason}\n")
    f.write(f"\n")
    f.write(f"Mean with children: {mean_with:.4f}\n")
    f.write(f"Mean without children: {mean_without:.4f}\n")
    f.write(f"Difference: {mean_diff:.4f}\n")
    f.write(f"T-statistic: {t_stat:.4f}\n")
    f.write(f"Chi-square: {chi2:.4f}\n")
    f.write(f"Cohen's d: {cohens_d:.4f}\n")

print("Analysis complete. Results saved to '_analysis_result.txt'")
