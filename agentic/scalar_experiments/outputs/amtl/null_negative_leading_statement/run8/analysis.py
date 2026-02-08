import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = 'amtl.csv'
df = pd.read_csv(csv_path)

# Clean/ensure categories
# Genus categories: Homo sapiens, Pan, Pongo, Papio
# Use Homo sapiens as reference? We'll create indicator for Homo vs others

df = df.copy()

# Create proportion response
# Guard against zero sockets
# If sockets == 0, drop (shouldn't happen per metadata)
df = df[df['sockets'] > 0]

# Encode categorical variables
# We'll use formula with C() to handle categories

# Build GLM binomial with logit, using successes and failures
# Use endog as proportion with weights = sockets
# Include age, prob_male, tooth_class, genus

formula = 'prop_amtl ~ C(genus) + age + prob_male + C(tooth_class)'

df['prop_amtl'] = df['num_amtl'] / df['sockets']

model = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), freq_weights=df['sockets'])
res = model.fit()

# Extract effect for Homo sapiens vs reference category
# Statsmodels uses first category alphabetically as reference by default
# We want effect of Homo sapiens vs non-human, so compute predicted difference

# Determine reference category
cats = df['genus'].astype('category').cat.categories

# Compute marginal effect: compare Homo sapiens vs average non-human at mean covariates

# Build a small dataframe for prediction at mean covariates
mean_age = df['age'].mean()
mean_prob_male = df['prob_male'].mean()

# Use most common tooth_class as reference for prediction (mode)
mode_tooth = df['tooth_class'].mode().iloc[0]

# Build prediction rows
rows = []
for genus in cats:
    rows.append({'genus': genus, 'age': mean_age, 'prob_male': mean_prob_male, 'tooth_class': mode_tooth})

pred_df = pd.DataFrame(rows)
preds = res.predict(pred_df)

# Compute Homo sapiens vs mean of non-human genera
if 'Homo sapiens' in list(cats):
    homo_pred = preds[list(cats).index('Homo sapiens')]
    non_human_preds = [preds[i] for i, g in enumerate(cats) if g != 'Homo sapiens']
    non_human_mean = float(np.mean(non_human_preds)) if non_human_preds else np.nan
    diff = homo_pred - non_human_mean
else:
    homo_pred = np.nan
    non_human_mean = np.nan
    diff = np.nan

# Extract coefficient for Homo sapiens if available
# Depending on reference, look for C(genus)[T.Homo sapiens]
coef = res.params.get('C(genus)[T.Homo sapiens]', np.nan)
se = res.bse.get('C(genus)[T.Homo sapiens]', np.nan)

# Simple z-test
if np.isfinite(coef) and np.isfinite(se) and se > 0:
    z = coef / se
    p = 2 * (1 - sm.stats.norm.cdf(abs(z)))
else:
    z = np.nan
    p = np.nan

print('Genus categories:', list(cats))
print('Homo pred at mean covariates:', homo_pred)
print('Non-human mean pred:', non_human_mean)
print('Difference (Homo - non-human mean):', diff)
print('Coef for Homo sapiens:', coef)
print('SE:', se)
print('z:', z)
print('p:', p)
print('\nModel summary:')
print(res.summary())
