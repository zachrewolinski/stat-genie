import pandas as pd
import statsmodels.api as sm

# Load data
df = pd.read_csv('crofoot.csv')

# Derived variables
# Relative group size (positive means focal larger)
df['rel_size'] = df['n_focal'] - df['n_other']
# Contest location advantage: positive if focal is closer to its home range center
# (smaller distance from center indicates closer to home)
df['loc_adv'] = df['dist_other'] - df['dist_focal']

# Logistic regression
X = df[['rel_size', 'loc_adv']]
X = sm.add_constant(X)
y = df['win']

model = sm.Logit(y, X)
res = model.fit(disp=False)

# Save a brief summary to a file for inspection if needed
with open('analysis_summary.txt', 'w') as f:
    f.write(res.summary2().as_text())

# Also compute simple correlations for reference
corr_rel_size = df['win'].corr(df['rel_size'])
corr_loc_adv = df['win'].corr(df['loc_adv'])

with open('analysis_stats.txt', 'w') as f:
    f.write(f"corr(win, rel_size) = {corr_rel_size}\n")
    f.write(f"corr(win, loc_adv) = {corr_loc_adv}\n")

# Print key stats
print(res.params)
print(res.pvalues)
