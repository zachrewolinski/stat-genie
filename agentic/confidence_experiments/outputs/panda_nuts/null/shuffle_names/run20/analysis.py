import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Map columns to meaning based on metadata + observed values
# age -> age in years (1-22)
# nuts_opened -> sex (f/m)
# seconds -> received help (y/N)
# help -> number of nuts opened
# chimpanzee -> duration seconds

# Clean help flag
help_col = df['seconds'].astype(str).str.strip()
# Normalize to lowercase
help_col = help_col.str.lower()
# Map yes/no
help_flag = help_col.map({'y': 1, 'yes': 1, 'n': 0, 'no': 0})

# Some values might be 'N' or 'n' etc
if help_flag.isna().any():
    # Try to map any non-mapped to 1 if startswith y
    help_flag = help_col.apply(lambda x: 1 if x.startswith('y') else 0)

# Build analysis dataframe
analysis_df = pd.DataFrame({
    'age': df['age'],
    'sex': df['nuts_opened'].astype(str).str.lower(),
    'helped': help_flag.astype(int),
    'nuts_opened': df['help'].astype(float),
    'seconds': df['chimpanzee'].astype(float)
})

# Drop rows with non-positive seconds
analysis_df = analysis_df.replace([np.inf, -np.inf], np.nan).dropna()
analysis_df = analysis_df[analysis_df['seconds'] > 0]

# Efficiency
analysis_df['efficiency'] = analysis_df['nuts_opened'] / analysis_df['seconds']

# Encode sex categorical
analysis_df['sex'] = analysis_df['sex'].replace({'f': 'f', 'm': 'm'})

# Descriptive stats
print('N rows:', len(analysis_df))
print('Sex counts:\n', analysis_df['sex'].value_counts())
print('Helped counts:\n', analysis_df['helped'].value_counts())
print('Efficiency summary:\n', analysis_df['efficiency'].describe())

# OLS on efficiency
ols_model = smf.ols('efficiency ~ age + C(sex) + helped', data=analysis_df).fit()
print('\nOLS summary (efficiency):')
print(ols_model.summary())

# Poisson with offset for rate
# To avoid issues with zero nuts, use Poisson on counts with log(seconds) offset
poisson_model = smf.glm(
    'nuts_opened ~ age + C(sex) + helped',
    data=analysis_df,
    family=sm.families.Poisson(),
    offset=np.log(analysis_df['seconds'])
).fit()
print('\nPoisson summary (rate via offset):')
print(poisson_model.summary())

# Negative binomial if overdispersion
nb_model = smf.glm(
    'nuts_opened ~ age + C(sex) + helped',
    data=analysis_df,
    family=sm.families.NegativeBinomial(alpha=1.0),
    offset=np.log(analysis_df['seconds'])
).fit()
print('\nNegative Binomial summary (rate via offset):')
print(nb_model.summary())

# Overdispersion check using Pearson chi2 / df
pearson_chi2 = poisson_model.pearson_chi2
ratio = pearson_chi2 / poisson_model.df_resid
print('\nOverdispersion ratio (Poisson Pearson chi2 / df):', ratio)

# Compute effect sizes as rate ratios (exp coefficients) for GLM models
coef = poisson_model.params
conf = poisson_model.conf_int()
rate_ratios = np.exp(coef)
rr_ci = np.exp(conf)
print('\nPoisson rate ratios:')
for term in coef.index:
    print(term, rate_ratios[term], rr_ci.loc[term].tolist(), poisson_model.pvalues[term])
