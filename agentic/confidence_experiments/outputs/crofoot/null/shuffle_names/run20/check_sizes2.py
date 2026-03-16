import pandas as pd

df = pd.read_csv('crofoot.csv')

# Suppose f_other is females in focal, dist_focal is males in focal
# Suppose win is females in other, focal is males in other
focal_total = df['f_other'] + df['dist_focal']
other_total = df['win'] + df['focal']

print('focal_total summary', focal_total.describe())
print('other_total summary', other_total.describe())

# See if totals are in plausible range 7-17 etc

