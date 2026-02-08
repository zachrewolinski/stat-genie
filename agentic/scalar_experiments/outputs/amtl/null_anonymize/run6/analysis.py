import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Rename for clarity
_df = _df.rename(columns={
    'feature1': 'tooth_class',
    'feature2': 'specimen_id',
    'feature3': 'missing',
    'feature4': 'observable',
    'feature5': 'age',
    'feature6': 'age_uncert',
    'feature7': 'sex',
    'feature8': 'genus',
    'feature9': 'region',
})

# Basic cleaning: drop rows with missing required fields
req_cols = ['missing', 'observable', 'age', 'sex', 'tooth_class', 'genus']
df = _df.dropna(subset=req_cols).copy()

# Ensure counts are valid
# Remove any rows with zero observable sockets (can't score AMTL)
df = df[df['observable'] > 0]
# Remove inconsistent rows where missing exceeds observable
df = df[df['missing'] <= df['observable']]

# Create Homo indicator (1 if Homo sapiens)
df['is_homo'] = (df['genus'] == 'Homo sapiens').astype(int)

# Expand to individual socket-level outcomes to avoid binomial weight issues
rows = []
for _, row in df.iterrows():
    total = int(row['observable'])
    miss = int(row['missing'])
    present = total - miss
    if total <= 0 or miss < 0 or present < 0:
        continue
    if miss > 0:
        r = row[['is_homo', 'age', 'sex', 'tooth_class']].to_dict()
        r['outcome'] = 1
        rows.append(pd.DataFrame([r] * miss))
    if present > 0:
        r = row[['is_homo', 'age', 'sex', 'tooth_class']].to_dict()
        r['outcome'] = 0
        rows.append(pd.DataFrame([r] * present))

expanded = pd.concat(rows, ignore_index=True)

formula = 'outcome ~ is_homo + age + sex + C(tooth_class)'
model = smf.glm(
    formula=formula,
    data=expanded,
    family=sm.families.Binomial()
).fit()

# Extract homo coefficient and p-value
coef = model.params.get('is_homo', np.nan)
pval = model.pvalues.get('is_homo', np.nan)

# Compute predicted AMTL rate for homo vs non-homo at mean covariates
mean_age = df['age'].mean()
mean_sex = df['sex'].mean()
# Use most common tooth class for baseline
mode_tooth = df['tooth_class'].mode().iloc[0]

pred_df = pd.DataFrame({
    'is_homo': [0, 1],
    'age': [mean_age, mean_age],
    'sex': [mean_sex, mean_sex],
    'tooth_class': [mode_tooth, mode_tooth],
})

pred = model.predict(pred_df)

non_homo_rate, homo_rate = pred.iloc[0], pred.iloc[1]

# Map evidence to Likert scale
# Use sign of effect, p-value strength, and magnitude of difference

diff = homo_rate - non_homo_rate

# Start with base magnitude from p-value
if np.isnan(pval):
    base = 0
elif pval < 0.001:
    base = 80
elif pval < 0.01:
    base = 65
elif pval < 0.05:
    base = 50
elif pval < 0.1:
    base = 30
else:
    base = 10

# Scale by effect size magnitude (rate difference)
# Cap scaling to avoid extremes from small p-values
scale = min(1.5, max(0.5, abs(diff) / 0.05))  # 5 percentage points as moderate
score = int(round(base * scale))

# Apply sign based on direction
if diff < 0:
    score = -score
elif diff == 0:
    score = 0

# Clip to [-100, 100]
score = max(-100, min(100, score))

# Write conclusion
with open('conclusion.txt', 'w') as f:
    f.write(str(score))

# Save a brief results summary for debugging (not requested, but helpful)
with open('analysis_summary.txt', 'w') as f:
    f.write(model.summary().as_text())
    f.write('\n\n')
    f.write(f'homo_rate={homo_rate:.6f}\n')
    f.write(f'non_homo_rate={non_homo_rate:.6f}\n')
    f.write(f'diff={diff:.6f}\n')
    f.write(f'coef={coef:.6f}\n')
    f.write(f'pval={pval:.6g}\n')
    f.write(f'score={score}\n')
