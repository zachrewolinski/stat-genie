import json
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'amtl.csv'
df = pd.read_csv(path)

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

# Create binary indicator for modern humans
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Response as proportion with binomial variance; use weights for number of trials
# Add a tiny offset to avoid 0/0 if any (should not happen given data description)
df['miss_prop'] = df['missing_teeth'] / df['observable_sockets']

# Fit GLM binomial with controls
formula = 'miss_prop ~ is_human + age + sex + C(tooth_class)'
model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Binomial(),
    var_weights=df['observable_sockets']
)
result = model.fit()

coef = result.params['is_human']
pval = result.pvalues['is_human']

# Determine response and scale
if coef > 0 and pval < 0.05:
    response = 'Yes'
    scale = round(50 + 50 * min(1.0, (0.05 - pval) / 0.05))
elif pval < 0.05:
    response = 'No'
    scale = round(50 - 50 * min(1.0, (0.05 - pval) / 0.05))
else:
    response = 'No'
    scale = round(50 - 50 * min(1.0, (pval - 0.05) / 0.95))

# Clamp to [0, 100]
scale = max(0, min(100, int(scale)))

output = {'response': response, 'scale': scale}

with open('conclusion.txt', 'w') as f:
    json.dump(output, f)
