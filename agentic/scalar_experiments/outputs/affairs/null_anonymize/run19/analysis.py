import pandas as pd
import numpy as np
from scipy import stats as st

# Load data
df = pd.read_csv('affairs.csv')

# Define engagement in affairs
engaged = df['feature2'] > 0

df = df.assign(engaged=engaged)

# Groups by children
mask_yes = df['feature6'] == 'yes'
mask_no = df['feature6'] == 'no'

n_yes = int(mask_yes.sum())
n_no = int(mask_no.sum())

p_yes = df.loc[mask_yes, 'engaged'].mean()
p_no = df.loc[mask_no, 'engaged'].mean()

diff = p_no - p_yes  # positive means more affairs without children

# Z-test for difference in proportions
p_pool = (p_yes * n_yes + p_no * n_no) / (n_yes + n_no)
se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_yes + 1 / n_no))
if se == 0:
    z = 0.0
else:
    z = diff / se

# Mean frequency difference check
mean_yes = df.loc[mask_yes, 'feature2'].mean()
mean_no = df.loc[mask_no, 'feature2'].mean()
mean_diff = mean_no - mean_yes

# Score mapping: z=3 -> 100
score = (z / 3) * 100
score = float(np.clip(score, -100, 100))

# If mean diff disagrees in sign with proportion diff, dampen confidence
if (diff == 0 and mean_diff != 0) or (diff != 0 and np.sign(diff) != np.sign(mean_diff)):
    score *= 0.5

score_int = int(np.round(score))

with open('conclusion.txt', 'w') as f:
    f.write(str(score_int))

# Print brief diagnostics for our own check
print({
    'n_yes': n_yes,
    'n_no': n_no,
    'p_yes': p_yes,
    'p_no': p_no,
    'diff': diff,
    'z': z,
    'mean_yes': mean_yes,
    'mean_no': mean_no,
    'mean_diff': mean_diff,
    'score': score,
    'score_int': score_int,
})
