import pandas as pd
import numpy as np
from scipy import stats

_df = pd.read_csv('reading.csv')

# compute reading speed as words per minute (feature20 equals this)
_df['speed_wpm'] = _df['feature20']

# dyslexia definition
_df['dyslexia_any'] = (_df['feature17'] == 1) | (_df['feature12'] > 0)
sub = _df[_df['dyslexia_any']].copy()

# per participant mean speed per condition
agg = sub.groupby(['feature1','feature3'])['speed_wpm'].mean().unstack()
paired = agg.dropna()

# compute descriptive stats
mean_on = paired[1].mean()
mean_off = paired[0].mean()
median_on = paired[1].median()
median_off = paired[0].median()

# paired t-test and CI
res = stats.ttest_rel(paired[1], paired[0])
diff = paired[1] - paired[0]
mean_diff = diff.mean()
std_diff = diff.std(ddof=1)
se_diff = std_diff / np.sqrt(len(diff))
# 95% CI
ci_low, ci_high = stats.t.interval(0.95, df=len(diff)-1, loc=mean_diff, scale=se_diff)

print(f"paired participants: {len(paired)}")
print(f"mean speed wpm on: {mean_on:.2f}, off: {mean_off:.2f}")
print(f"median speed wpm on: {median_on:.2f}, off: {median_off:.2f}")
print(f"mean diff (on-off): {mean_diff:.2f} wpm")
print(f"95% CI: [{ci_low:.2f}, {ci_high:.2f}]")
print(f"t-test: t={res.statistic:.3f}, p={res.pvalue:.3f}")

# effect size (paired Cohen's d)
cohen_d = mean_diff / std_diff
print(f"paired Cohen d: {cohen_d:.3f}")

# counts per condition
counts = sub['feature3'].value_counts().sort_index()
print("obs per condition:", counts.to_dict())
