import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Define nut-cracking efficiency as nuts opened per minute
# (feature5 = nuts opened, feature6 = duration seconds)
df['efficiency_per_min'] = (df['feature5'] / df['feature6']) * 60.0

# Clean any potential infinities (should not exist given min duration > 0)
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['efficiency_per_min'])

# Rename for clarity
clean = df.rename(columns={
    'feature2': 'age',
    'feature3': 'sex',
    'feature7': 'help'
})

# Fit linear model with categorical sex and help
model = smf.ols('efficiency_per_min ~ age + C(sex) + C(help)', data=clean).fit(cov_type='HC3')

print('N:', len(clean))
print(model.summary())

# Extract p-values and coefficients for reporting
results = pd.DataFrame({
    'coef': model.params,
    'pvalue': model.pvalues
})
print('\nCoefficients and p-values:')
print(results)

# Save key results for potential inspection
results.to_csv('analysis_results.csv')
