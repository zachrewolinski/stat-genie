import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('panda_nuts.csv')

# Map variables based on metadata + observed values
# age: column 'age'
# sex: column 'nuts_opened' (values f/m)
# received help: column 'seconds' (values y/N)
# nuts opened (count): column 'help'
# session duration seconds: column 'chimpanzee'

# Clean/encode
analysis_df = df.copy()
# Rename to avoid duplicate column names
analysis_df = analysis_df.rename(columns={
    'nuts_opened': 'sex',          # f/m
    'sex': 'hammer_type',          # wood/Q/G/etc (not used)
    'help': 'nuts_opened',         # count
    'chimpanzee': 'duration_sec',  # seconds
    'seconds': 'received_help'     # Y/N
})

# Standardize help coding to boolean
analysis_df['received_help'] = analysis_df['received_help'].astype(str).str.upper().str.strip()
analysis_df['received_help'] = analysis_df['received_help'].map({'Y': 1, 'N': 0})

# Sex coding
analysis_df['sex'] = analysis_df['sex'].astype(str).str.lower().str.strip()
analysis_df['sex'] = analysis_df['sex'].map({'f': 1, 'm': 0})

# Efficiency: nuts per second
analysis_df['efficiency'] = analysis_df['nuts_opened'] / analysis_df['duration_sec']

# Drop rows with missing
analysis_df = analysis_df.dropna(subset=['age', 'sex', 'received_help', 'nuts_opened', 'duration_sec', 'efficiency'])

# OLS with robust SE
ols_model = smf.ols('efficiency ~ age + sex + received_help', data=analysis_df).fit(cov_type='HC3')

# Poisson regression for counts with log(duration) offset
# Add small epsilon to duration to avoid log(0) (shouldn't be zero)
analysis_df['log_duration'] = np.log(analysis_df['duration_sec'])
poisson_model = smf.glm('nuts_opened ~ age + sex + received_help',
                        data=analysis_df,
                        family=sm.families.Poisson(),
                        offset=analysis_df['log_duration']).fit(cov_type='HC3')

# Summaries
print('N=', len(analysis_df))
print('\nOLS robust summary:')
print(ols_model.summary())
print('\nPoisson (rate) robust summary:')
print(poisson_model.summary())

# Also compute simple group means for context
mean_by_sex = analysis_df.groupby('sex')['efficiency'].mean()
mean_by_help = analysis_df.groupby('received_help')['efficiency'].mean()
print('\nMean efficiency by sex (0=male,1=female):')
print(mean_by_sex)
print('\nMean efficiency by received_help (0=no,1=yes):')
print(mean_by_help)

# Correlation with age
age_corr = analysis_df[['age', 'efficiency']].corr().iloc[0,1]
print('\nCorrelation age vs efficiency:', age_corr)
