import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

help_col = df['seconds'].astype(str).str.strip().str.lower()
help_flag = help_col.map({'y': 1, 'yes': 1, 'n': 0, 'no': 0})
if help_flag.isna().any():
    help_flag = help_col.apply(lambda x: 1 if x.startswith('y') else 0)

analysis_df = pd.DataFrame({
    'age': df['age'],
    'sex': df['nuts_opened'].astype(str).str.lower(),
    'helped': help_flag.astype(int),
    'nuts_opened': df['help'].astype(float),
    'seconds': df['chimpanzee'].astype(float)
}).replace([np.inf, -np.inf], np.nan).dropna()
analysis_df = analysis_df[analysis_df['seconds'] > 0]

analysis_df['efficiency'] = analysis_df['nuts_opened'] / analysis_df['seconds']
analysis_df['sex'] = analysis_df['sex'].replace({'f': 'f', 'm': 'm'})

# Negative binomial with estimated alpha via discrete model
# Use log(seconds) as offset to model rate
formula = 'nuts_opened ~ age + C(sex) + helped'

# Create design matrices
import patsy

y, X = patsy.dmatrices(formula, analysis_df, return_type='dataframe')
offset = np.log(analysis_df['seconds'])

nb2 = sm.NegativeBinomial(y, X, offset=offset)
nb2_res = nb2.fit(disp=False)

print(nb2_res.summary())

# Extract rate ratios and confidence intervals
params = nb2_res.params
conf = nb2_res.conf_int()
rr = np.exp(params)
rr_ci = np.exp(conf)

print('\nRate ratios (NB2):')
for term in params.index:
    print(term, rr[term], rr_ci.loc[term].tolist(), nb2_res.pvalues[term])

print('\nAlpha (overdispersion):', nb2_res.params.get('alpha', np.nan))
