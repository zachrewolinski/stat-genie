import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load dataset
csv_path = 'panda_nuts.csv'
df = pd.read_csv(csv_path)

# Basic cleaning: keep relevant columns, drop missing
# Define efficiency as nuts opened per second
# Add small epsilon to seconds to avoid divide by zero (shouldn't be zero)
df['efficiency'] = df['nuts_opened'] / df['seconds']

# Standardize categorical variables
# 'sex' is f/m, 'help' is y/N (yes/no)
df['sex'] = df['sex'].astype('category')
df['help'] = df['help'].astype('category')

# Drop rows with missing in key variables
analysis_df = df[['efficiency', 'age', 'sex', 'help']].dropna()

# Fit OLS model
model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=analysis_df).fit()

# Also compute simple group means for context
means_by_help = analysis_df.groupby('help')['efficiency'].mean()
means_by_sex = analysis_df.groupby('sex')['efficiency'].mean()

# Save key results
summary_text = model.summary().as_text()

with open('analysis_results.txt', 'w') as f:
    f.write(summary_text)
    f.write('\n\nGroup means by help:\n')
    f.write(means_by_help.to_string())
    f.write('\n\nGroup means by sex:\n')
    f.write(means_by_sex.to_string())

print(model.summary())
print('\nGroup means by help')
print(means_by_help)
print('\nGroup means by sex')
print(means_by_sex)
