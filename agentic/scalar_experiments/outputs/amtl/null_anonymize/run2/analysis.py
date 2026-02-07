import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('amtl.csv')

# Rename columns for clarity
col_map = {
    'feature1': 'tooth_class',
    'feature2': 'specimen_id',
    'feature3': 'missing_teeth',
    'feature4': 'observable_sockets',
    'feature5': 'age',
    'feature6': 'age_uncertainty',
    'feature7': 'sex',
    'feature8': 'genus',
    'feature9': 'region',
}

df = df.rename(columns=col_map)

# Drop rows with missing critical fields
needed = ['missing_teeth', 'observable_sockets', 'age', 'sex', 'tooth_class', 'genus']
missing_before = df[needed].isna().sum()

# Filter to valid rows
mask = df[needed].notna().all(axis=1)
# observable_sockets should be >0
mask &= df['observable_sockets'] > 0

clean = df.loc[mask].copy()

# Binomial response as proportion with weights
clean['prop_missing'] = clean['missing_teeth'] / clean['observable_sockets']

# Use categorical for genus and tooth_class
clean['genus'] = clean['genus'].astype('category')
clean['tooth_class'] = clean['tooth_class'].astype('category')

# Set reference genus to non-human by making Homo sapiens explicit ref? We want coefficient for Homo vs others.
# We'll set reference category as non-human aggregate by releveling with one non-human genus.
# Use Pan as reference and interpret Homo coefficient relative to Pan.
if 'Pan' in clean['genus'].cat.categories:
    clean['genus'] = clean['genus'].cat.reorder_categories(
        ['Pan'] + [g for g in clean['genus'].cat.categories if g != 'Pan'],
        ordered=False
    )

# Build model
formula = 'prop_missing ~ C(genus) + age + sex + C(tooth_class)'
model = smf.glm(
    formula=formula,
    data=clean,
    family=sm.families.Binomial(),
    var_weights=clean['observable_sockets']
).fit()

print('Rows used:', len(clean), 'of', len(df))
print('Genus categories:', list(clean['genus'].cat.categories))
print(model.summary())

# Extract Homo sapiens effect vs reference (Pan)
params = model.params
pvalues = model.pvalues

coef_name = None
for name in params.index:
    if 'C(genus)' in name and 'Homo sapiens' in name:
        coef_name = name
        break

if coef_name is None:
    raise SystemExit('Homo sapiens coefficient not found')

coef = params[coef_name]
pval = pvalues[coef_name]

# Compute odds ratio
oratio = float((coef).astype(float))

# Predict marginal means: compare Homo sapiens vs non-human average holding covariates
# We'll approximate by setting age and sex to means and tooth_class to distribution.
mean_age = clean['age'].mean()
mean_sex = clean['sex'].mean()

# Build data for each genus with average covariates and each tooth_class, weighted by observed tooth_class distribution
class_counts = clean['tooth_class'].value_counts(normalize=True)

rows = []
for genus in clean['genus'].cat.categories:
    for tooth_class, weight in class_counts.items():
        rows.append({
            'genus': genus,
            'age': mean_age,
            'sex': mean_sex,
            'tooth_class': tooth_class,
            'weight': weight,
        })

pred_df = pd.DataFrame(rows)

# Predict probability
pred_df['pred'] = model.predict(pred_df)

# Weighted mean by tooth_class distribution
mean_pred = pred_df.groupby('genus').apply(lambda g: (g['pred'] * g['weight']).sum())

print('\nAdjusted predicted AMTL proportion by genus (avg covariates):')
print(mean_pred.sort_values())

# Compare Homo sapiens to average of non-human genera
nonhuman = mean_pred.drop('Homo sapiens', errors='ignore')
if 'Homo sapiens' in mean_pred.index:
    homo_pred = mean_pred.loc['Homo sapiens']
    nonhuman_mean = nonhuman.mean()
    diff = homo_pred - nonhuman_mean
    print('\nHomo sapiens adjusted vs non-human mean diff:', diff)
    print('Homo sapiens adjusted prop:', homo_pred)
    print('Non-human mean prop:', nonhuman_mean)
    print('Homo sapiens coef (log-odds vs Pan):', coef)
    print('Homo sapiens p-value:', pval)
