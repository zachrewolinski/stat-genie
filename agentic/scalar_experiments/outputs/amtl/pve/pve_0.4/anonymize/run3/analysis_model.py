import pandas as pd
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('amtl.csv')

# Fit OLS with cluster-robust SEs by specimen id (feature2)
formula = 'feature3 ~ C(feature8) + feature5 + feature7 + C(feature1) + feature4'
model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['feature2']})

print(model.summary())

# Extract genus coefficients relative to Homo sapiens (reference)
params = model.params
pvalues = model.pvalues

# Display differences: other genera vs Homo
for genus in ['Pan', 'Pongo', 'Papio']:
    term = f'C(feature8)[T.{genus}]'
    if term in params:
        print(genus, 'coef', params[term], 'p', pvalues[term])

# Compute adjusted means per genus at mean covariates
mean_age = df['feature5'].mean()
mean_sex = df['feature7'].mean()
mean_socket = df['feature4'].mean()

# use Posterior as base? statsmodels uses first alphabetically? For C(feature1), default is alphabetical. Let's get categories
print('feature1 categories', df['feature1'].unique())

# We'll set tooth class to Posterior (most common) for prediction
pred_df = pd.DataFrame({
    'feature8': ['Homo sapiens','Pan','Pongo','Papio'],
    'feature5': mean_age,
    'feature7': mean_sex,
    'feature1': 'Posterior',
    'feature4': mean_socket,
})

pred = model.get_prediction(pred_df).summary_frame(alpha=0.05)
print(pd.concat([pred_df, pred], axis=1))
