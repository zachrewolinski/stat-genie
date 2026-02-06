import pandas as pd
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('panda_nuts.csv')

# Map shuffled columns to their actual meaning based on metadata
# age_years: column labeled 'hammer'
# sex: column labeled 'nuts_opened'
# help_received: column labeled 'seconds' (y/N)
# nuts_opened_count: column labeled 'help'
# session_seconds: column labeled 'chimpanzee'

df = pd.DataFrame({
    'age_years': _df['hammer'],
    'sex': _df['nuts_opened'],
    'help_received': _df['seconds'],
    'nuts_opened_count': _df['help'],
    'session_seconds': _df['chimpanzee'],
})

# Compute efficiency as nuts opened per second
# Avoid division by zero; session_seconds should be > 0 in this dataset

df['efficiency'] = df['nuts_opened_count'] / df['session_seconds']

# Clean help indicator
# Normalize help to categorical with levels 'N' and 'y'
df['help_received'] = df['help_received'].astype(str).str.strip()

# Basic sanity filters (drop any rows with missing or non-positive durations)
df = df[df['session_seconds'] > 0].dropna(subset=['efficiency', 'age_years', 'sex', 'help_received'])

# Fit linear model: efficiency ~ age + sex + help
model = smf.ols('efficiency ~ age_years + C(sex) + C(help_received)', data=df).fit(cov_type='HC3')

# Save key outputs for inspection if needed
summary = model.summary().as_text()
with open('analysis_summary.txt', 'w') as f:
    f.write(summary)

# Extract p-values for predictors
pvals = model.pvalues.to_dict()

# Store a small results table
results = []
for term in ['age_years', 'C(sex)[T.m]', 'C(help_received)[T.y]']:
    if term in pvals:
        results.append((term, pvals[term]))

results_df = pd.DataFrame(results, columns=['term', 'p_value'])
results_df.to_csv('analysis_pvalues.csv', index=False)

print(results_df)
