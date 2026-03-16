import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

csv_path = 'crofoot.csv'
df = pd.read_csv(csv_path)

# Alternative relative measures
# size ratio and difference

df['size_diff'] = df['feature7'] - df['feature8']
df['size_ratio'] = df['feature7'] / df['feature8']

# location difference and ratio (focal distance / other distance)
df['loc_diff'] = df['feature5'] - df['feature6']
df['loc_ratio'] = df['feature5'] / df['feature6']

# Also 'relative home advantage' = (other distance - focal distance), positive means focal closer to home center

df['home_adv'] = df['feature6'] - df['feature5']

outcomes = df['feature4']

predictor_sets = {
    'diffs': ['size_diff','loc_diff'],
    'ratios': ['size_ratio','loc_ratio'],
    'homeadv+size': ['size_diff','home_adv'],
    'raw sizes+dist': ['feature7','feature8','feature5','feature6'],
}

for name, cols in predictor_sets.items():
    X = sm.add_constant(df[cols])
    try:
        res = sm.Logit(outcomes, X).fit(disp=False)
    except Exception as e:
        print(name, 'failed', e)
        continue
    print('\nModel:', name)
    print(res.summary())
    print('Pseudo R2:', 1 - res.llf/res.llnull)

# Also check correlation between outcome and predictors (point-biserial)
for col in ['size_diff','size_ratio','loc_diff','loc_ratio','home_adv']:
    r, p = stats.pointbiserialr(outcomes, df[col])
    print(f"\nPoint-biserial {col}: r={r:.3f}, p={p:.4f}")
