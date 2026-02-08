import pandas as pd
import numpy as np
from scipy import stats

# Load data
_df = pd.read_csv('affairs.csv')

# Identify columns from metadata
# feature2: affair frequency; feature6: children yes/no

affairs = _df['feature2']
children = _df['feature6']

# Split groups
mask_yes = children.str.lower() == 'yes'
mask_no = children.str.lower() == 'no'

ay = affairs[mask_yes].astype(float)
an = affairs[mask_no].astype(float)

# Basic stats
mean_yes = ay.mean()
mean_no = an.mean()

# Cohen's d (pooled)
ny, nn = ay.size, an.size
sy, sn = ay.std(ddof=1), an.std(ddof=1)
sp = np.sqrt(((ny - 1) * sy**2 + (nn - 1) * sn**2) / (ny + nn - 2))
cohen_d = (mean_yes - mean_no) / sp

# Nonparametric test (Mann-Whitney) and t-test
mw = stats.mannwhitneyu(ay, an, alternative='two-sided')
tt = stats.ttest_ind(ay, an, equal_var=False)

# Proportion with any affairs (>0)
prop_yes = (ay > 0).mean()
prop_no = (an > 0).mean()

# Simple scoring: negative d supports decrease.
# Scale by effect size and significance strength.
# Convert d to support for decrease: support = -d
support = -cohen_d

# Weight by p-values: use smaller of MW and t-test for evidence
p = min(mw.pvalue, tt.pvalue)
if p < 0.001:
    sig_w = 1.0
elif p < 0.01:
    sig_w = 0.85
elif p < 0.05:
    sig_w = 0.7
elif p < 0.1:
    sig_w = 0.55
else:
    sig_w = 0.4

# Map support to score with tanh for bounded output
score = 100 * np.tanh(support * 1.5) * sig_w

# Clamp and round to int
score_int = int(np.clip(np.rint(score), -100, 100))

# Save details for inspection
with open('analysis_summary.txt', 'w') as f:
    f.write(f"n_yes={ny}, n_no={nn}\n")
    f.write(f"mean_yes={mean_yes:.4f}, mean_no={mean_no:.4f}\n")
    f.write(f"prop_yes={prop_yes:.4f}, prop_no={prop_no:.4f}\n")
    f.write(f"cohen_d={cohen_d:.4f}\n")
    f.write(f"mw_p={mw.pvalue:.6g}, tt_p={tt.pvalue:.6g}\n")
    f.write(f"score={score:.4f}, score_int={score_int}\n")

with open('conclusion.txt', 'w') as f:
    f.write(str(score_int))
