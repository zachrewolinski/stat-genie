import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'amtl.csv'
df = pd.read_csv(path)

# Map shuffled columns to actual meanings based on info.json
# Actual variables
# tooth_class: Anterior/Posterior/Premolar stored in 'sockets'
# specimen_id: stored in 'prob_male'
# num_amtl: stored in 'genus'
# num_sockets: stored in 'age'
# age: stored in 'pop'
# stdev_age: stored in 'num_amtl'
# prob_male: stored in 'stdev_age'
# genus: stored in 'tooth_class'
# population: stored in 'specimen'

df = df.rename(columns={
    'sockets': 'tooth_class',
    'prob_male': 'specimen_id',
    'genus': 'num_amtl',
    'age': 'num_sockets',
    'pop': 'age',
    'num_amtl': 'age_sd',
    'stdev_age': 'prob_male',
    'tooth_class': 'genus',
    'specimen': 'population'
})

# Basic checks
# Ensure counts are integers where expected
# num_amtl and num_sockets are counts

# Drop any rows with missing or invalid counts
# Also ensure num_amtl <= num_sockets

# Convert counts to numeric
for col in ['num_amtl', 'num_sockets']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove rows with missing or impossible values
valid = df['num_amtl'].notna() & df['num_sockets'].notna() & (df['num_sockets'] > 0) & (df['num_amtl'] >= 0)
df = df.loc[valid].copy()

# Some data may have fractional values if data were averaged; allow but cap to num_sockets
# If num_amtl exceeds num_sockets due to rounding, drop those rows

df = df.loc[df['num_amtl'] <= df['num_sockets']].copy()

# Create human indicator

df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Prepare response for binomial GLM
# Use proportion with weights = num_sockets

df['amtl_rate'] = df['num_amtl'] / df['num_sockets']

# Fit GLM: amtl_rate ~ is_human + age + prob_male + tooth_class
# Use binomial with var_weights as num_sockets

model = smf.glm(
    formula='amtl_rate ~ is_human + age + prob_male + C(tooth_class)',
    data=df,
    family=sm.families.Binomial(),
    var_weights=df['num_sockets']
).fit()

# Extract effect for is_human
coef = model.params['is_human']
se = model.bse['is_human']
pval = model.pvalues['is_human']

# Odds ratio and 95% CI
or_val = np.exp(coef)
ci_low = np.exp(coef - 1.96*se)
ci_high = np.exp(coef + 1.96*se)

print('N rows used:', len(df))
print('is_human coefficient:', coef)
print('SE:', se)
print('p-value:', pval)
print('Odds ratio:', or_val)
print('95% CI:', (ci_low, ci_high))

# Also compute genus-specific adjusted predictions at mean age/prob_male and for tooth_class overall (average)
# For interpretability, compute predicted rate for human vs non-human (set is_human=0/1) with average covariates

mean_age = df['age'].mean()
mean_prob_male = df['prob_male'].mean()

# Compute average predicted rate across tooth classes (by weighting with observed distribution)
classes = df['tooth_class'].unique()
class_weights = df['tooth_class'].value_counts(normalize=True)

preds = {}
for is_h in [0, 1]:
    pred_rates = []
    weights = []
    for cls in classes:
        row = pd.DataFrame({
            'is_human': [is_h],
            'age': [mean_age],
            'prob_male': [mean_prob_male],
            'tooth_class': [cls]
        })
        rate = model.predict(row)[0]
        pred_rates.append(rate)
        weights.append(class_weights[cls])
    preds[is_h] = float(np.sum(np.array(pred_rates) * np.array(weights)))

print('Predicted AMTL rate (non-human):', preds[0])
print('Predicted AMTL rate (human):', preds[1])

# Save key outputs for downstream use

out = pd.DataFrame({
    'coef': [coef],
    'se': [se],
    'pval': [pval],
    'or': [or_val],
    'ci_low': [ci_low],
    'ci_high': [ci_high],
    'pred_nonhuman': [preds[0]],
    'pred_human': [preds[1]],
    'n': [len(df)]
})

out.to_csv('model_results.csv', index=False)
