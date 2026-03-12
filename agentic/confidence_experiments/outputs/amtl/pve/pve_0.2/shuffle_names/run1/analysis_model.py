import pandas as pd
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('amtl.csv')

# Set explicit reference levels for clarity
# - species/genus (tooth_class): Homo sapiens as reference
# - tooth class (sockets): Posterior as reference
formula = (
    'genus ~ C(tooth_class, Treatment(reference="Homo sapiens")) '
    '+ pop + stdev_age '
    '+ C(sockets, Treatment(reference="Posterior"))'
)

model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['prob_male']})

print(model.summary())

# Adjusted predictions at mean covariates, Posterior socket class
mean_pop = df['pop'].mean()
mean_sex = df['stdev_age'].mean()
ref_socket = 'Posterior'

species = df['tooth_class'].unique()

pred_df = pd.DataFrame({
    'tooth_class': species,
    'pop': mean_pop,
    'stdev_age': mean_sex,
    'sockets': ref_socket,
})

pred = model.get_prediction(pred_df)
summary = pred.summary_frame(alpha=0.05)

pred_df = pred_df.copy()
pred_df['pred_mean'] = summary['mean']
pred_df['mean_se'] = summary['mean_se']

print('\nAdjusted predictions at mean covariates (socket=Posterior):')
print(pred_df)

# Report coefficient p-values for genus differences vs Homo sapiens
print('\nCoefficients for genus vs Homo sapiens:')
for term in model.params.index:
    if term.startswith('C(tooth_class') and 'Homo sapiens' not in term:
        print(term, 'coef=', model.params[term], 'p=', model.pvalues[term])
