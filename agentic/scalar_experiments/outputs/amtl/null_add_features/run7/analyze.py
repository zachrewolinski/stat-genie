import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Keep relevant columns
cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class']

df = _df[cols].copy()

# Drop missing or invalid rows
for c in cols:
    df = df[df[c].notna()]

# Ensure valid counts
# num_amtl should be <= sockets and sockets > 0

df = df[(df['sockets'] > 0) & (df['num_amtl'] >= 0) & (df['num_amtl'] <= df['sockets'])]

# Ensure categorical types

df['genus'] = df['genus'].astype('category')
df['tooth_class'] = df['tooth_class'].astype('category')

# Set reference category for genus to Homo sapiens if present
if 'Homo sapiens' in df['genus'].cat.categories:
    df['genus'] = df['genus'].cat.reorder_categories(
        ['Homo sapiens'] + [g for g in df['genus'].cat.categories if g != 'Homo sapiens'],
        ordered=False,
    )

# Fit binomial GLM with counts
# Use endog as [success, failure]
endog = np.column_stack([df['num_amtl'], df['sockets'] - df['num_amtl']])

formula = 'num_amtl + (sockets - num_amtl) ~ C(genus) + age + prob_male + C(tooth_class)'
model = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
res = model.fit()

# Compute marginal predicted AMTL proportion for each genus
# Use the observed covariates and tooth_class for each row, but set genus to target and average

def mean_pred_for_genus(target_genus: str) -> float:
    temp = df.copy()
    temp['genus'] = target_genus
    pred = res.predict(temp)
    return float(np.mean(pred))

# Collect genus list

genera = list(df['genus'].cat.categories)

pred_means = {g: mean_pred_for_genus(g) for g in genera}

# Compare Homo sapiens to others
homo = 'Homo sapiens'
if homo not in pred_means:
    # If missing, fallback to max genus name for reference
    homo = genera[0]

comparisons = {}
for g in genera:
    if g == homo:
        continue
    comparisons[g] = pred_means[homo] - pred_means[g]

# Compute approximate z-scores for genus coefficients vs Homo using model params
# Coef interpretation: log-odds difference of non-Homo vs Homo.
# Negative coef => non-Homo lower than Homo.

coef_info = {}
for g in genera:
    if g == homo:
        continue
    key = f'C(genus)[T.{g}]'
    if key in res.params:
        coef = res.params[key]
        se = res.bse[key]
        z = coef / se if se > 0 else np.nan
        coef_info[g] = (coef, se, z)

# Build scalar decision
# Heuristic: look at mean differences and significance direction.

# Determine if Homo has higher predicted AMTL than each other genus
higher_all = all(diff > 0 for diff in comparisons.values()) if comparisons else False

# Determine strength from average difference magnitude
mean_diff = float(np.mean(list(comparisons.values()))) if comparisons else 0.0

# Use coefficient z-scores for direction; want non-Homo negative (Homo higher)
neg_z = []
for g, (coef, se, z) in coef_info.items():
    if np.isfinite(z):
        neg_z.append(-z)  # positive means Homo higher

avg_z = float(np.mean(neg_z)) if neg_z else 0.0

# Map to scalar
# Start at 0, then adjust based on mean diff and avg_z.
# mean_diff ~ probability difference; scale 0.01 => 5 points.
scalar = 0.0

scalar += mean_diff * 500.0  # 0.10 diff => 50 points

# z-score scaling: avg_z 2 => +20
scalar += avg_z * 10.0

# If not higher for all genera, dampen
if comparisons and not higher_all:
    scalar *= 0.4

# Clamp and round to int
scalar = int(np.clip(np.round(scalar), -100, 100))

# Write conclusion
with open('conclusion.txt', 'w') as f:
    f.write(str(scalar))

# Also save brief results for debugging (not requested, but useful if needed)
with open('analysis_summary.txt', 'w') as f:
    f.write('Predicted mean AMTL proportion by genus:\n')
    for g, v in pred_means.items():
        f.write(f'{g}: {v:.4f}\n')
    f.write('\nDifferences (Homo - other):\n')
    for g, v in comparisons.items():
        f.write(f'{g}: {v:.4f}\n')
    f.write('\nGenus coefficient info (non-Homo vs Homo):\n')
    for g, (coef, se, z) in coef_info.items():
        f.write(f'{g}: coef={coef:.4f}, se={se:.4f}, z={z:.3f}\n')
    f.write(f'\nScalar: {scalar}\n')
