import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
amtl = pd.read_csv('amtl.csv')

# Prepare variables
amtl['is_human'] = (amtl['genus'] == 'Homo sapiens').astype(int)

# Basic descriptive stats
summary = amtl.groupby('genus')['num_amtl'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)

# Regression: num_amtl ~ is_human + age + prob_male + tooth_class
# Use robust (HC3) standard errors for heteroskedasticity
model = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class)', data=amtl).fit(cov_type='HC3')

coef = model.params['is_human']
se = model.bse['is_human']
pval = model.pvalues['is_human']

# Predicted adjusted means for human vs nonhuman at average covariates
mean_age = amtl['age'].mean()
mean_prob_male = amtl['prob_male'].mean()
# For tooth_class, use observed distribution to compute marginal mean
classes = amtl['tooth_class'].unique()

# Build design matrix manually by predicting for each tooth_class and averaging by observed weights
weights = amtl['tooth_class'].value_counts(normalize=True)

preds = {}
for is_human in [0, 1]:
    pred = 0.0
    for cls, w in weights.items():
        df = pd.DataFrame({
            'is_human': [is_human],
            'age': [mean_age],
            'prob_male': [mean_prob_male],
            'tooth_class': [cls]
        })
        pred += w * model.predict(df)[0]
    preds[is_human] = pred

adj_diff = preds[1] - preds[0]

# Save results for inspection
print('Descriptive means by genus (num_amtl):')
print(summary)
print('\nRegression coefficient for is_human:')
print({'coef': coef, 'se': se, 'pval': pval})
print('\nAdjusted mean (human):', preds[1])
print('Adjusted mean (nonhuman):', preds[0])
print('Adjusted difference (human - nonhuman):', adj_diff)

# Also fit model with genus categorical to compare Homo vs each genus
model_genus = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=amtl).fit(cov_type='HC3')
print('\nGenus model coefficients:')
print(model_genus.params.filter(like='C(genus)'))
print(model_genus.pvalues.filter(like='C(genus)'))

# Export some key stats for downstream use
results = {
    'coef_is_human': coef,
    'se_is_human': se,
    'pval_is_human': pval,
    'adj_diff': adj_diff,
    'adj_mean_human': preds[1],
    'adj_mean_nonhuman': preds[0],
    'summary_means': summary.to_dict()
}

pd.Series(results).to_json('analysis_results.json')
