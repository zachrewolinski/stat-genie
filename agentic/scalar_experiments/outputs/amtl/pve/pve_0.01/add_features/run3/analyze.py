import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('amtl.csv')

print('shape', df.shape)
print('columns', df.columns.tolist())

# Basic checks
print(df[['num_amtl','sockets','age','prob_male','tooth_class','genus']].head())
print('num_amtl stats', df['num_amtl'].describe())
print('sockets stats', df['sockets'].describe())

# Check if num_amtl is integer-like
print('num_amtl unique sample', df['num_amtl'].head(10).tolist())

# If num_amtl and sockets are both numeric, check if num_amtl within [0, sockets]
within = ((df['num_amtl'] >= 0) & (df['num_amtl'] <= df['sockets'])).mean()
print('num_amtl within 0..sockets proportion', within)

# Check per genus counts
print(df['genus'].value_counts())

# Attempt binomial regression if plausible counts

# Use only relevant columns and drop missing
sub = df[['num_amtl','sockets','age','prob_male','tooth_class','genus']].copy()
sub = sub.dropna()

# Determine if num_amtl is integer (within tolerance)
sub['num_amtl_round'] = np.round(sub['num_amtl'])
print('fraction near integer', np.mean(np.isclose(sub['num_amtl'], sub['num_amtl_round'], atol=1e-6)))

# Fit two models: OLS on num_amtl (if not binomial), and Binomial GLM if counts valid

# Encode genus as categorical with reference non-human? We'll set Homo sapiens as level for comparison

# Statsmodels formula uses C(genus) etc.

# OLS model
ols = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=sub).fit()
print(ols.summary())

# Binomial model only if num_amtl between 0 and sockets and integer
if within > 0.95 and np.mean(np.isclose(sub['num_amtl'], sub['num_amtl_round'], atol=1e-6)) > 0.95:
    sub['num_amtl_int'] = sub['num_amtl_round'].astype(int)
    # Use GLM binomial with freq weights? We'll use endog as proportion with weights
    sub['prop'] = sub['num_amtl_int'] / sub['sockets']
    glm = smf.glm('prop ~ C(genus) + age + prob_male + C(tooth_class)',
                  data=sub,
                  family=sm.families.Binomial(),
                  freq_weights=sub['sockets']).fit()
    print(glm.summary())
else:
    print('Skipping binomial model; num_amtl not counts within sockets')

