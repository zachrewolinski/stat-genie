import csv
import math
from collections import defaultdict

# Read the dataset
with open('affairs.csv', 'r') as f:
    reader = csv.DictReader(f)
    data = list(reader)

print("=" * 80)
print("ANALYSIS: Does having children decrease engagement in extramarital affairs?")
print("=" * 80)
print()

# Basic dataset information
print(f"Total observations: {len(data)}")
print()

# Separate data by children status
children_yes = []
children_no = []
children_yes_affairs = []
children_no_affairs = []

for row in data:
    affair_freq = float(row['feature2'])
    if row['feature6'] == 'yes':
        children_yes.append(row)
        children_yes_affairs.append(affair_freq)
    else:
        children_no.append(row)
        children_no_affairs.append(affair_freq)

print(f"Respondents with children: {len(children_yes)}")
print(f"Respondents without children: {len(children_no)}")
print()

# Calculate statistics
def mean(values):
    return sum(values) / len(values) if values else 0

def median(values):
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n == 0:
        return 0
    if n % 2 == 0:
        return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
    else:
        return sorted_values[n//2]

def std_dev(values):
    if len(values) < 2:
        return 0
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)

# Analyze affair engagement by children status
print("-" * 80)
print("AFFAIR ENGAGEMENT BY CHILDREN STATUS")
print("-" * 80)
print()

mean_affairs_with_children = mean(children_yes_affairs)
mean_affairs_without_children = mean(children_no_affairs)
median_affairs_with_children = median(children_yes_affairs)
median_affairs_without_children = median(children_no_affairs)
std_with_children = std_dev(children_yes_affairs)
std_without_children = std_dev(children_no_affairs)

print(f"With children:")
print(f"  Mean affair frequency: {mean_affairs_with_children:.4f}")
print(f"  Median affair frequency: {median_affairs_with_children:.4f}")
print(f"  Std deviation: {std_with_children:.4f}")
print()

print(f"Without children:")
print(f"  Mean affair frequency: {mean_affairs_without_children:.4f}")
print(f"  Median affair frequency: {median_affairs_without_children:.4f}")
print(f"  Std deviation: {std_without_children:.4f}")
print()

# Calculate percentage engaged in affairs (any affair > 0)
count_affairs_with_children = sum(1 for x in children_yes_affairs if x > 0)
count_affairs_without_children = sum(1 for x in children_no_affairs if x > 0)
pct_affairs_with_children = (count_affairs_with_children / len(children_yes_affairs)) * 100
pct_affairs_without_children = (count_affairs_without_children / len(children_no_affairs)) * 100

print(f"Percentage engaged in affairs (frequency > 0):")
print(f"  With children: {pct_affairs_with_children:.2f}% ({count_affairs_with_children}/{len(children_yes_affairs)})")
print(f"  Without children: {pct_affairs_without_children:.2f}% ({count_affairs_without_children}/{len(children_no_affairs)})")
print()

# Chi-square test for affair occurrence (yes/no)
print("-" * 80)
print("STATISTICAL TESTS")
print("-" * 80)
print()

# Contingency table: children (yes/no) vs had affair (yes/no)
# [children=yes, affair=no], [children=yes, affair=yes]
# [children=no, affair=no], [children=no, affair=yes]
a = len(children_yes_affairs) - count_affairs_with_children  # children=yes, affair=no
b = count_affairs_with_children                               # children=yes, affair=yes
c = len(children_no_affairs) - count_affairs_without_children # children=no, affair=no
d = count_affairs_without_children                            # children=no, affair=yes

print("Contingency Table:")
print(f"                      No Affair    Had Affair    Total")
print(f"With children         {a:8d}     {b:9d}    {a+b:6d}")
print(f"Without children      {c:8d}     {d:9d}    {c+d:6d}")
print(f"Total                 {a+c:8d}     {b+d:9d}    {a+b+c+d:6d}")
print()

# Chi-square calculation
n = a + b + c + d
expected_a = (a + b) * (a + c) / n
expected_b = (a + b) * (b + d) / n
expected_c = (c + d) * (a + c) / n
expected_d = (c + d) * (b + d) / n

chi2 = ((a - expected_a)**2 / expected_a +
        (b - expected_b)**2 / expected_b +
        (c - expected_c)**2 / expected_c +
        (d - expected_d)**2 / expected_d)

# For 2x2 table, df = 1
# Critical value at p=0.05 for df=1 is 3.841
critical_value = 3.841

print(f"Chi-square test for independence:")
print(f"  Chi-square statistic: {chi2:.4f}")
print(f"  Degrees of freedom: 1")
print(f"  Critical value (p=0.05): {critical_value:.4f}")
if chi2 > critical_value:
    print(f"  Result: Statistically significant association (chi2 > {critical_value:.4f})")
    p_significant = True
else:
    print(f"  Result: No statistically significant association (chi2 <= {critical_value:.4f})")
    p_significant = False
print()

# Two-sample t-test
def t_test(group1, group2):
    n1 = len(group1)
    n2 = len(group2)
    m1 = mean(group1)
    m2 = mean(group2)

    # Calculate pooled standard deviation
    var1 = sum((x - m1)**2 for x in group1) / (n1 - 1) if n1 > 1 else 0
    var2 = sum((x - m2)**2 for x in group2) / (n2 - 1) if n2 > 1 else 0

    pooled_se = math.sqrt(var1/n1 + var2/n2)

    if pooled_se == 0:
        return 0, False

    t_stat = (m1 - m2) / pooled_se

    # For large samples (n > 30), critical value at p=0.05 (two-tailed) is approximately 1.96
    critical_t = 1.96
    significant = abs(t_stat) > critical_t

    return t_stat, significant

t_stat, t_significant = t_test(children_yes_affairs, children_no_affairs)

print(f"Independent t-test:")
print(f"  T-statistic: {t_stat:.4f}")
print(f"  Critical value (p=0.05, two-tailed): ±1.96")
if t_significant:
    print(f"  Result: Statistically significant difference (|t| > 1.96)")
else:
    print(f"  Result: No statistically significant difference (|t| <= 1.96)")
print()

# Summary of findings
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

difference = mean_affairs_without_children - mean_affairs_with_children
pct_difference = pct_affairs_without_children - pct_affairs_with_children

print(f"Mean difference (no children - with children): {difference:.4f}")
print(f"Percentage point difference: {pct_difference:.2f}%")
print()

if mean_affairs_with_children < mean_affairs_without_children:
    print("FINDING: People WITH children have LOWER average affair frequency")
    print(f"         than those WITHOUT children.")
    direction = "decrease"
else:
    print("FINDING: People WITH children have HIGHER OR EQUAL average affair frequency")
    print(f"         than those WITHOUT children.")
    direction = "increase or no change"
print()

if p_significant or t_significant:
    print("STATISTICAL SIGNIFICANCE: The difference is statistically significant")
    print(f"                          (at least one test shows significance)")
    significance = "significant"
else:
    print("STATISTICAL SIGNIFICANCE: The difference is NOT statistically significant")
    print(f"                          (tests do not show strong evidence)")
    significance = "not significant"
print()

print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()

if mean_affairs_with_children < mean_affairs_without_children:
    print("Having children is associated with DECREASED engagement in extramarital affairs.")
    if p_significant or t_significant:
        print("This difference is statistically significant.")
    else:
        print("However, the difference is not statistically significant at the 0.05 level.")
else:
    print("Having children is NOT associated with decreased engagement in extramarital affairs.")
    print("The data shows no decrease (or possibly an increase) in affair frequency.")
