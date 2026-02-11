import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'amtl.csv'
df = pd.read_csv(path)

# Rename for clarity
# feature1: tooth class
# feature3: missing count
# feature4: observable sockets
# feature5: age
# feature7: sex estimate
# feature8: genus

# Filter rows with valid counts
# Ensure non-negative and observable > 0
mask = (df['feature4'] > 0) & (df['feature3'] >= 0)
df = df.loc[mask].copy()

# Create success/failure counts for binomial GLM
# success = missing, failure = observable - missing
# If missing > observable, clamp? assume data consistent
fail = df['feature4'] - df['feature3']
# Drop any negative failures (data issues)
valid = fail >= 0
if not valid.all():
    df = df.loc[valid].copy()
    fail = df['feature4'] - df['feature3']

# Indicator for Homo sapiens vs others
# Keep original genus for reference
# We'll model with Homo indicator + other covariates

df['is_human'] = (df['feature8'] == 'Homo sapiens').astype(int)

# Build design matrix using patsy
# Use tooth class categorical, sex numeric, age numeric
# We include is_human and also allow tooth class categorical

# Prepare endog as 2-column successes/failures
endog = np.column_stack([df['feature3'].astype(int).values, fail.astype(int).values])

# Formula for covariates
formula = 'is_human + feature5 + feature7 + C(feature1)'

# Build exog with patsy via GLM from formula
model = sm.GLM(endog, sm.add_constant(pd.get_dummies(df[['is_human','feature5','feature7','feature1']], columns=['feature1'], drop_first=True)),
               family=sm.families.Binomial())

result = model.fit()

# Extract p-value and coefficient for is_human
coef = result.params['is_human']
pval = result.pvalues['is_human']

# Determine response and scale
# Yes if p<0.05 and coef>0 (higher AMTL)
# Scale based on p-value magnitude and direction
if (pval < 0.05) and (coef > 0):
    response = 'Yes'
    # Map p-value to 50-100 scale, stronger evidence higher
    # Use -log10(p) capped
    strength = min(-np.log10(pval), 10)
    scale = int(round(50 + (strength/10)*50))
else:
    response = 'No'
    # If coef <=0 or p not sig, scale below 50
    # Use p-value and sign to set strength
    if coef <= 0:
        # More negative -> stronger no
        strength = min(abs(coef), 2.0) / 2.0
        scale = int(round(50 - strength*50))
    else:
        strength = 1 - min(pval/0.5, 1)  # p near 0.5 => weak
        scale = int(round(50 - strength*50))

# Ensure bounds 0-100
scale = max(0, min(100, scale))

# Write conclusion
out = {'response': response, 'scale': int(scale)}
with open('conclusion.txt', 'w') as f:
    json.dump(out, f)

print(json.dumps({"coef_is_human": coef, "pval_is_human": pval, "n": len(df)}))
