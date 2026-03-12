import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
csv_path = "hurricane.csv"
df = pd.read_csv(csv_path)

# Define variables
# Femininity rating from expert coders
fem = df['feature4']
# Alternate femininity from MTurk
fem_alt = df['feature12']
# Gender indicator
fem_bin = df['feature6']

# Outcome: deaths
# log( deaths + 1 ) to handle zeros
log_deaths = np.log1p(df['feature8'])

# Controls for intensity
df = df.copy()
df['log_deaths'] = log_deaths

# Check simple correlations
corr_fem = stats.pearsonr(fem, log_deaths)
corr_fem_alt = stats.pearsonr(fem_alt, log_deaths)
corr_fem_bin = stats.pearsonr(fem_bin, log_deaths)

print("Correlation fem vs log_deaths:", corr_fem)
print("Correlation fem_alt vs log_deaths:", corr_fem_alt)
print("Correlation fem_bin vs log_deaths:", corr_fem_bin)

# Regression models
# Model 1: log_deaths ~ fem
model1 = smf.ols('log_deaths ~ feature4', data=df).fit()

# Model 2: log_deaths ~ fem + intensity controls
# Use wind speed, pressure, category, year
model2 = smf.ols('log_deaths ~ feature4 + feature13 + feature5 + feature7 + feature2', data=df).fit()

# Model 3: log_deaths ~ fem_alt + intensity controls
model3 = smf.ols('log_deaths ~ feature12 + feature13 + feature5 + feature7 + feature2', data=df).fit()

# Model 4: log_deaths ~ fem_bin + intensity controls
model4 = smf.ols('log_deaths ~ feature6 + feature13 + feature5 + feature7 + feature2', data=df).fit()

print("\nModel1 summary (fem only)")
print(model1.summary())

print("\nModel2 summary (fem + controls)")
print(model2.summary())

print("\nModel3 summary (fem_alt + controls)")
print(model3.summary())

print("\nModel4 summary (fem_bin + controls)")
print(model4.summary())

