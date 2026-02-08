import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy

# Load data
_df = pd.read_csv('amtl.csv')

# Rename for clarity
_df = _df.rename(columns={
    'feature1': 'tooth_class',
    'feature2': 'specimen_id',
    'feature3': 'missing',
    'feature4': 'observable',
    'feature5': 'age',
    'feature6': 'age_uncertainty',
    'feature7': 'sex',
    'feature8': 'genus',
    'feature9': 'region',
})

# Basic validity filters
_df = _df.dropna(subset=['missing', 'observable', 'age', 'sex', 'tooth_class', 'genus'])
_df = _df[_df['observable'] > 0]
_df = _df[_df['missing'] >= 0]
_df = _df[_df['missing'] <= _df['observable']]

# Define successes/failures
_df = _df.copy()
_df['successes'] = _df['missing']
_df['failures'] = _df['observable'] - _df['missing']

# Set reference category for genus and tooth class
# Ensure Homo sapiens is reference
_df['genus'] = _df['genus'].astype('category')
_df['genus'] = _df['genus'].cat.set_categories(
    ['Homo sapiens', 'Pan', 'Papio', 'Pongo'],
    ordered=False
)

# Some datasets might use 'Premolar' etc; keep as categorical
_df['tooth_class'] = _df['tooth_class'].astype('category')

# Build design matrices for binomial GLM
formula = 'successes + failures ~ C(genus, Treatment(reference="Homo sapiens")) + age + sex + C(tooth_class)'

y, X = patsy.dmatrices(formula, _df, return_type='dataframe')

model = sm.GLM(y, X, family=sm.families.Binomial())
result = model.fit()

# Contrast: Homo sapiens vs average of non-human genera
param_names = list(result.params.index)
L = np.zeros(len(param_names))

# Coefficients for non-human genera in Treatment coding
coeffs = {
    'C(genus, Treatment(reference="Homo sapiens"))[T.Pan]': 1/3,
    'C(genus, Treatment(reference="Homo sapiens"))[T.Papio]': 1/3,
    'C(genus, Treatment(reference="Homo sapiens"))[T.Pongo]': 1/3,
}

for i, name in enumerate(param_names):
    if name in coeffs:
        L[i] = coeffs[name]

# L * params gives average non-human effect vs Homo (log-odds)
# We want Homo - avg_nonhuman, so use -L
contrast = result.t_test(-L)
contrast_est = float(np.asarray(contrast.effect).ravel()[0])
contrast_p = float(contrast.pvalue)

# Predicted probability at mean age/sex and overall tooth class distribution
mean_age = _df['age'].mean()
mean_sex = _df['sex'].mean()

# Use overall tooth class distribution as weights
class_counts = _df['tooth_class'].value_counts(normalize=True)

# Build a small dataframe for prediction
rows = []
for genus in ['Homo sapiens', 'Pan', 'Papio', 'Pongo']:
    for tooth_class, weight in class_counts.items():
        rows.append({
            'genus': genus,
            'age': mean_age,
            'sex': mean_sex,
            'tooth_class': tooth_class,
            'weight': weight,
        })

pred_df = pd.DataFrame(rows)

design_info = X.design_info
X_pred = patsy.build_design_matrices([design_info], pred_df, return_type='dataframe')[0]

pred_lin = result.predict(X_pred)

pred_df['pred_prob'] = pred_lin

# Weighted average per genus
weighted = pred_df.groupby('genus').apply(
    lambda g: np.sum(g['pred_prob'] * g['weight']) / np.sum(g['weight'])
)

homo_prob = float(weighted.loc['Homo sapiens'])
nonhuman_prob = float(weighted.loc[['Pan', 'Papio', 'Pongo']].mean())

prob_diff = homo_prob - nonhuman_prob

# Map to Likert scale [-100, 100]
# Effect size: 0.20 probability diff => 70 points
# Significance: p<=0.01 => +30 points, p>=0.5 => +0
sign = 1 if prob_diff > 0 else (-1 if prob_diff < 0 else 0)

effect_points = min(70.0, abs(prob_diff) / 0.20 * 70.0)

if contrast_p <= 0.01:
    sig_points = 30.0
elif contrast_p >= 0.5:
    sig_points = 0.0
else:
    # Linear interpolation between 0.5 and 0.01
    sig_points = (0.5 - contrast_p) / (0.5 - 0.01) * 30.0

score = sign * min(100.0, effect_points + sig_points)
score_int = int(round(score))

with open('conclusion.txt', 'w') as f:
    f.write(str(score_int))

# Save a compact summary for review if needed
summary = {
    'n': int(len(_df)),
    'homo_prob': homo_prob,
    'nonhuman_prob': nonhuman_prob,
    'prob_diff': prob_diff,
    'contrast_log_odds_diff': contrast_est,
    'contrast_pvalue': contrast_p,
    'score_int': score_int,
}

pd.DataFrame([summary]).to_csv('analysis_summary.csv', index=False)
