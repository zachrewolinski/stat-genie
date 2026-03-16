import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Clean/prepare
# Standardize categorical variables
df['sex'] = df['sex'].astype('category')
df['help'] = df['help'].astype('category')

# Efficiency: nuts opened per second
# Avoid division by zero (seconds min 2.5 per metadata)
df['rate'] = df['nuts_opened'] / df['seconds']

# Cluster groups for repeated measures
cluster_groups = df['chimpanzee']

# Base linear model on rate
model_rate = smf.ols('rate ~ age + C(sex) + C(help)', data=df).fit()

# Poisson rate model with offset log(seconds)
# Use nuts_opened as count, offset log(seconds)
# For zero seconds (none), skip
poisson_model = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=df,
                        family=sm.families.Poisson(),
                        offset=np.log(df['seconds'])).fit()

# Poisson with cluster-robust SEs
poisson_cluster = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=df,
                          family=sm.families.Poisson(),
                          offset=np.log(df['seconds'])).fit(cov_type='cluster',
                                                            cov_kwds={'groups': cluster_groups})

# Check overdispersion: Pearson chi2 / df
pearson_chi2 = sum(poisson_model.resid_pearson**2)
overdisp_ratio = pearson_chi2 / poisson_model.df_resid

# Negative binomial as robustness
nb_model = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=df,
                   family=sm.families.NegativeBinomial(alpha=1.0),
                   offset=np.log(df['seconds'])).fit()

# Negative binomial with cluster-robust SEs
nb_cluster = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=df,
                     family=sm.families.NegativeBinomial(alpha=1.0),
                     offset=np.log(df['seconds'])).fit(cov_type='cluster',
                                                       cov_kwds={'groups': cluster_groups})

# Cluster-robust SEs by individual chimpanzee (repeated measures)
model_rate_cluster = model_rate.get_robustcov_results(cov_type='cluster', groups=cluster_groups)

# Output key results
print('N rows:', len(df))
print('\nLinear rate model (OLS) summary coefficients:')
print(model_rate.summary().tables[1])
print('\nLinear rate model with cluster-robust SEs (by chimpanzee):')
print(model_rate_cluster.summary().tables[1])

print('\nPoisson rate model coefficients:')
print(poisson_model.summary().tables[1])
print('Overdispersion ratio (Pearson chi2/df):', overdisp_ratio)
print('\nPoisson rate model with cluster-robust SEs (by chimpanzee):')
print(poisson_cluster.summary().tables[1])

print('\nNegative binomial model coefficients:')
print(nb_model.summary().tables[1])
print('\nNegative binomial model with cluster-robust SEs (by chimpanzee):')
print(nb_cluster.summary().tables[1])

# Additionally compute simple group comparisons
# Mean rates by sex/help
print('\nMean rate by sex:')
print(df.groupby('sex')['rate'].mean())
print('\nMean rate by help:')
print(df.groupby('help')['rate'].mean())

# Correlation age-rate
corr = df['age'].corr(df['rate'])
print('\nCorrelation age vs rate:', corr)
