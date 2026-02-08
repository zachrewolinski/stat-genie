import pandas as pd
import numpy as np
from scipy import stats

# Load data

df = pd.read_csv('affairs.csv')

# Identify columns based on value patterns
# affairs count in past year is in column 'age' per value coding (0,1,2,3,7,12)
# children indicator is in column 'religiousness' (yes/no)

affairs = df['age']
children = df['religiousness']

# Ensure categories
children_yes = affairs[children == 'yes']
children_no = affairs[children == 'no']

# Mean counts
mean_yes = children_yes.mean()
mean_no = children_no.mean()

# Proportion any affairs (>0)
prop_any_yes = (children_yes > 0).mean()
prop_any_no = (children_no > 0).mean()

# Difference in means and proportions
mean_diff = mean_no - mean_yes  # positive means children associated with fewer affairs
prop_diff = prop_any_no - prop_any_yes

# Effect size (Cohen's d) for counts
# Use pooled std with ddof=1
n1 = len(children_no)
n2 = len(children_yes)
std1 = children_no.std(ddof=1)
std2 = children_yes.std(ddof=1)
pooled = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2)) if (n1+n2-2) > 0 else np.nan
cohens_d = (mean_no - mean_yes) / pooled if pooled and pooled > 0 else np.nan

# t-test
# Use Welch's t-test due to variance difference
welch = stats.ttest_ind(children_no, children_yes, equal_var=False)

# For proportion any affairs, use two-proportion z-test approx
p1 = prop_any_no
p2 = prop_any_yes
p_pool = (p1*n1 + p2*n2) / (n1+n2)
se = np.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2)) if p_pool*(1-p_pool) > 0 else np.nan
z = (p1 - p2) / se if se and se > 0 else np.nan
p_z = 2*(1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan

print('n_no', n1, 'n_yes', n2)
print('mean_no', mean_no, 'mean_yes', mean_yes, 'mean_diff(no-yes)', mean_diff)
print('prop_any_no', prop_any_no, 'prop_any_yes', prop_any_yes, 'prop_diff', prop_diff)
print('cohens_d', cohens_d)
print('welch_t', welch.statistic, 'p', welch.pvalue)
print('z', z, 'p', p_z)
