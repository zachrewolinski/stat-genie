import pandas as pd
import statsmodels.api as sm

# Load data
DF_PATH = 'panda_nuts.csv'
df = pd.read_csv(DF_PATH)

# Column meanings inferred from metadata mismatch:
# - 'hammer' appears to be age in years
# - 'nuts_opened' appears to be sex (m/f)
# - 'seconds' appears to be help from another chimp (y/n)
# - 'help' appears to be number of nuts opened in session
# - 'chimpanzee' appears to be session duration in seconds

# Build variables
sec = df['seconds'].astype(str).str.lower()
df['help_yes'] = sec.map({'y': 1, 'n': 0})

df['sex_m'] = df['nuts_opened'].map({'m': 1, 'f': 0})

# Efficiency: nuts opened per second
# Guard against zero duration (not expected, but safe)
df['efficiency'] = df['help'] / df['chimpanzee'].replace({0: pd.NA})

# Drop rows with missing derived values
analysis_df = df.dropna(subset=['efficiency', 'help_yes', 'sex_m', 'hammer']).copy()

# Regression: efficiency ~ age (hammer) + sex + help
X = analysis_df[['hammer', 'sex_m', 'help_yes']]
X = sm.add_constant(X)
y = analysis_df['efficiency']

model = sm.OLS(y, X).fit()

print('Model summary:')
print(model.summary())

# Group means for interpretability
print('\nGroup means (efficiency):')
print(analysis_df.groupby('sex_m')['efficiency'].mean().rename({0: 'female', 1: 'male'}))
print(analysis_df.groupby('help_yes')['efficiency'].mean().rename({0: 'no_help', 1: 'help'}))

