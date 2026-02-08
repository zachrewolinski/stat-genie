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

# Basic cleaning
_df = _df.dropna(subset=['missing', 'observable', 'age', 'sex', 'tooth_class', 'genus'])
_df = _df[_df['observable'] > 0]
_df = _df[_df['missing'] >= 0]
_df = _df[_df['missing'] <= _df['observable']]

_df['is_human'] = (_df['genus'] == 'Homo sapiens').astype(int)

# Construct response as successes/failures
_df['failures'] = _df['observable'] - _df['missing']

# GLM binomial with counts
# Use tooth_class as categorical, adjust for age and sex
_formula = 'missing + failures ~ is_human + age + sex + C(tooth_class)'

model = smf.glm(
    formula=_formula,
    data=_df,
    family=sm.families.Binomial()
)
res = model.fit()

# Extract effect for is_human
coef = res.params.get('is_human', np.nan)
se = res.bse.get('is_human', np.nan)
p = res.pvalues.get('is_human', np.nan)

# Odds ratio
or_val = float(np.exp(coef)) if np.isfinite(coef) else np.nan

# Compute predicted marginal difference at mean covariates
# Create a representative row at mean values and reference category
mean_age = _df['age'].mean()
mean_sex = _df['sex'].mean()
# Use the most frequent tooth_class as reference for prediction rows
ref_tooth = _df['tooth_class'].value_counts().index[0]

base = {
    'age': mean_age,
    'sex': mean_sex,
    'tooth_class': ref_tooth,
    'is_human': 0,
}

human = dict(base)
human['is_human'] = 1

pred_df = pd.DataFrame([base, human])

pred = res.predict(pred_df)

# Difference in predicted probability
pred_diff = float(pred[1] - pred[0])

# Map to Likert scale (-100..100)
# Heuristic: sign from coef; magnitude from effect size and p-value
# Start with base magnitude from odds ratio and predicted difference
mag = 0.0
if np.isfinite(or_val):
    # OR effect scaling
    if or_val >= 3:
        mag += 50
    elif or_val >= 2:
        mag += 35
    elif or_val >= 1.5:
        mag += 20
    elif or_val >= 1.2:
        mag += 10
    elif or_val >= 1.0:
        mag += 0
    elif or_val >= 0.83:
        mag -= 10
    elif or_val >= 0.67:
        mag -= 20
    elif or_val >= 0.5:
        mag -= 35
    else:
        mag -= 50

# Add contribution from predicted probability difference
if np.isfinite(pred_diff):
    if abs(pred_diff) >= 0.20:
        mag += 25 * np.sign(pred_diff)
    elif abs(pred_diff) >= 0.10:
        mag += 15 * np.sign(pred_diff)
    elif abs(pred_diff) >= 0.05:
        mag += 8 * np.sign(pred_diff)
    elif abs(pred_diff) >= 0.02:
        mag += 4 * np.sign(pred_diff)

# Add confidence based on p-value
if np.isfinite(p):
    if p < 1e-4:
        mag += 15 * np.sign(coef)
    elif p < 1e-3:
        mag += 12 * np.sign(coef)
    elif p < 1e-2:
        mag += 8 * np.sign(coef)
    elif p < 0.05:
        mag += 4 * np.sign(coef)
    elif p < 0.1:
        mag += 2 * np.sign(coef)
    else:
        mag += 0

# If coef negative, flip sign appropriately
if np.isfinite(coef):
    if coef < 0 and mag > 0:
        mag = -mag
    elif coef > 0 and mag < 0:
        mag = -mag

# Clamp to [-100, 100] and round to nearest integer
scalar = int(np.clip(np.round(mag), -100, 100))

# Write conclusion
with open('conclusion.txt', 'w') as f:
    f.write(str(scalar))

# Save some diagnostics for potential inspection
with open('analysis_summary.txt', 'w') as f:
    f.write('Rows used: %d\n' % len(_df))
    f.write('is_human coef: %.6f\n' % coef)
    f.write('is_human OR: %.6f\n' % or_val)
    f.write('is_human p: %.6g\n' % p)
    f.write('predicted diff (human - non-human): %.6f\n' % pred_diff)
    f.write('scalar: %d\n' % scalar)

