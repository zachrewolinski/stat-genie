import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
info = json.load(open('info.json'))

df = pd.read_csv('amtl.csv')

# Rename columns for clarity
col_map = {
    'feature1': 'tooth_class',
    'feature2': 'specimen_id',
    'feature3': 'missing',
    'feature4': 'observable',
    'feature5': 'age',
    'feature6': 'age_uncertainty',
    'feature7': 'sex',
    'feature8': 'genus',
    'feature9': 'region',
}

df = df.rename(columns=col_map)

# Drop rows with missing key fields or invalid counts
needed = ['missing', 'observable', 'age', 'sex', 'tooth_class', 'genus']
clean = df.dropna(subset=needed).copy()

# Ensure counts are numeric
clean['missing'] = pd.to_numeric(clean['missing'], errors='coerce')
clean['observable'] = pd.to_numeric(clean['observable'], errors='coerce')
clean['age'] = pd.to_numeric(clean['age'], errors='coerce')
clean['sex'] = pd.to_numeric(clean['sex'], errors='coerce')

clean = clean.dropna(subset=['missing','observable','age','sex'])

# Remove impossible rows
clean = clean[(clean['observable'] > 0) & (clean['missing'] >= 0) & (clean['missing'] <= clean['observable'])]

# Create failures column
clean['present'] = clean['observable'] - clean['missing']

# Set categorical order: use non-human primates as baseline (Pan)
# We'll set genus with Homo sapiens as one level; baseline = Pan
clean['genus'] = clean['genus'].astype('category')

# Ensure tooth_class categorical
clean['tooth_class'] = clean['tooth_class'].astype('category')

# Build GLM binomial with successes/failures
# Model: missing vs present ~ genus + age + sex + tooth_class
formula = 'missing + present ~ C(genus) + age + sex + C(tooth_class)'
model = smf.glm(formula=formula, data=clean, family=sm.families.Binomial()).fit()

# Extract effect for Homo sapiens vs baseline genus
params = model.params
bse = model.bse
pvalues = model.pvalues

# Find coefficient for C(genus)[T.Homo sapiens]
coef_key = None
for key in params.index:
    if key.startswith('C(genus)') and 'Homo sapiens' in key:
        coef_key = key
        break

result = {
    'n_rows': int(clean.shape[0]),
    'coef_key': coef_key,
    'coef': float(params[coef_key]) if coef_key else None,
    'se': float(bse[coef_key]) if coef_key else None,
    'p': float(pvalues[coef_key]) if coef_key else None,
}

# Also compute marginal predicted difference at mean covariates for Homo vs average non-human
# Build two profiles at mean age/sex and reference tooth_class (most common)
mean_age = float(clean['age'].mean())
mean_sex = float(clean['sex'].mean())
ref_tooth = clean['tooth_class'].value_counts().idxmax()

# Build dataframes for prediction across genera
base = pd.DataFrame({
    'age': [mean_age],
    'sex': [mean_sex],
    'tooth_class': [ref_tooth],
})

# Use one row per genus
pred_rows = []
for g in clean['genus'].cat.categories:
    row = base.copy()
    row['genus'] = g
    pred_rows.append(row)

pred_df = pd.concat(pred_rows, ignore_index=True)

pred = model.predict(pred_df)

pred_map = {g: float(p) for g, p in zip(clean['genus'].cat.categories, pred)}

# Compute difference Homo sapiens vs average of non-human
non_human = [g for g in clean['genus'].cat.categories if g != 'Homo sapiens']
mean_non_human = float(np.mean([pred_map[g] for g in non_human])) if non_human else None
homo_pred = pred_map.get('Homo sapiens', None)

result.update({
    'pred_homo': homo_pred,
    'pred_non_human_mean': mean_non_human,
    'pred_diff': (homo_pred - mean_non_human) if homo_pred is not None and mean_non_human is not None else None
})

with open('analysis_results.json','w') as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
