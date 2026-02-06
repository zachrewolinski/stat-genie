import pandas as pd
import statsmodels.formula.api as smf

# Load data
csv_path = 'panda_nuts.csv'
df = pd.read_csv(csv_path)

# Define efficiency: nuts opened per second
# Avoid division by zero just in case
_df = df.copy()
_df['efficiency'] = _df['feature5'] / _df['feature6']

# Rename columns for clarity
_df = _df.rename(columns={
    'feature2': 'age',
    'feature3': 'sex',
    'feature7': 'help'
})

# Fit OLS model with categorical sex/help
model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=_df).fit()

# Print summary for inspection
print(model.summary())

# Save key results for later use if needed
results = model.params.to_frame('coef')
results['pvalue'] = model.pvalues
results.to_csv('analysis_results.csv')
