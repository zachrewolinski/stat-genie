import pandas as pd
import numpy as np
from scipy import stats

# Load data
path = 'affairs.csv'
df = pd.read_csv(path)

# Normalize children column
# Expect 'yes'/'no'
df = df.copy()

df['children'] = df['children'].astype(str).str.strip().str.lower()

# Basic counts
counts = df['children'].value_counts(dropna=False)

# Affairs variable
# Use numeric

aff = pd.to_numeric(df['affairs'], errors='coerce')

# Split groups
mask_yes = df['children'] == 'yes'
mask_no = df['children'] == 'no'

aff_yes = aff[mask_yes].dropna()
aff_no = aff[mask_no].dropna()

# Means
mean_yes = aff_yes.mean()
mean_no = aff_no.mean()

# Proportion with any affairs (>0)
prop_yes = (aff_yes > 0).mean()
prop_no = (aff_no > 0).mean()

# Two-sample t-test (unequal variances)
# Note: affairs is discrete/skewed, but t-test for mean difference

t_stat, t_p = stats.ttest_ind(aff_yes, aff_no, equal_var=False)

# Mann-Whitney U test
u_stat, u_p = stats.mannwhitneyu(aff_yes, aff_no, alternative='two-sided')

# Effect size: Cohen's d for mean difference
# Pooled std for unequal size
n_yes = len(aff_yes)
n_no = len(aff_no)
std_yes = aff_yes.std(ddof=1)
std_no = aff_no.std(ddof=1)

# Use pooled sd
pooled_sd = np.sqrt(((n_yes-1)*std_yes**2 + (n_no-1)*std_no**2) / (n_yes+n_no-2))
cohen_d = (mean_yes - mean_no) / pooled_sd if pooled_sd > 0 else np.nan

# Proportion difference test (two-proportion z-test)
# Use normal approx
p1 = prop_yes
p2 = prop_no
n1 = n_yes
n2 = n_no
p_pool = ((aff_yes > 0).sum() + (aff_no > 0).sum()) / (n1 + n2)
se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
if se > 0:
    z = (p1 - p2) / se
    p_z = 2 * (1 - stats.norm.cdf(abs(z)))
else:
    z = np.nan
    p_z = np.nan

# Also compute odds ratio for any affair (yes vs no)
# Build 2x2

a = (aff_yes > 0).sum()  # yes children, affair
b = (aff_yes == 0).sum() # yes children, no affair
c = (aff_no > 0).sum()   # no children, affair
d = (aff_no == 0).sum()  # no children, no affair

# Add 0.5 continuity if needed
if min(a,b,c,d) == 0:
    a_c,b_c,c_c,d_c = a+0.5,b+0.5,c+0.5,d+0.5
else:
    a_c,b_c,c_c,d_c = a,b,c,d

odds_ratio = (a_c * d_c) / (b_c * c_c)

# Print summary
print('Counts children:', counts.to_dict())
print('N yes:', n_yes, 'N no:', n_no)
print('Mean affairs yes:', mean_yes, 'no:', mean_no, 'diff (yes-no):', mean_yes-mean_no)
print('Prop any affairs yes:', prop_yes, 'no:', prop_no, 'diff:', prop_yes-prop_no)
print('t-test p:', t_p)
print('Mann-Whitney p:', u_p)
print('Cohen d:', cohen_d)
print('Two-proportion z p:', p_z)
print('Odds ratio (any affair, yes vs no):', odds_ratio)
