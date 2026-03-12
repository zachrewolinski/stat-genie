import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv('amtl.csv')

# Ensure categorical baselines
cat_genus = ['Homo sapiens', 'Pan', 'Pongo', 'Papio']
cat_tooth = ['Anterior', 'Posterior', 'Premolar']

df['genus'] = pd.Categorical(df['genus'], categories=cat_genus)
df['tooth_class'] = pd.Categorical(df['tooth_class'], categories=cat_tooth)

df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Model with genus categories
formula = 'num_amtl ~ C(genus) + age + prob_male + C(tooth_class)'
model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['specimen']})

# Model with human vs non-human
formula2 = 'num_amtl ~ is_human + age + prob_male + C(tooth_class)'
model2 = smf.ols(formula2, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['specimen']})

# Summaries
print('N:', len(df))
print('Model with genus categories (Homo baseline)')
print(model.summary().tables[1])

# Extract genus contrasts
params = model.params
conf = model.conf_int()

genus_effects = {}
for g in ['Pan', 'Pongo', 'Papio']:
    key = f'C(genus)[T.{g}]'
    if key in params:
        genus_effects[g] = {
            'coef': float(params[key]),
            'ci_low': float(conf.loc[key, 0]),
            'ci_high': float(conf.loc[key, 1]),
            'pvalue': float(model.pvalues[key]),
        }

print('\nGenus effects (non-human minus Homo sapiens):')
for g, v in genus_effects.items():
    print(g, v)

# Human vs non-human
print('\nModel with is_human')
print(model2.summary().tables[1])

coef = float(model2.params['is_human'])
ci = model2.conf_int().loc['is_human'].tolist()
pval = float(model2.pvalues['is_human'])
print('\nHuman vs non-human effect:')
print({'coef': coef, 'ci_low': float(ci[0]), 'ci_high': float(ci[1]), 'pvalue': pval})

# Provide mean outcomes by genus (raw)
print('\nRaw mean num_amtl by genus:')
print(df.groupby('genus')['num_amtl'].mean())

# Compute adjusted means at average covariates for each genus
mean_age = df['age'].mean()
mean_prob_male = df['prob_male'].mean()
# choose reference tooth class (Anterior) then compute differences using model
# We'll compute predicted means for each genus across tooth classes weighted by observed distribution

# Create design rows for each genus and tooth_class with mean covariates
rows = []
for g in cat_genus:
    for tc in cat_tooth:
        rows.append({'genus': g, 'tooth_class': tc, 'age': mean_age, 'prob_male': mean_prob_male})

pred_df = pd.DataFrame(rows)
pred_df['pred'] = model.predict(pred_df)

# Weight by observed tooth_class distribution
weights = df['tooth_class'].value_counts(normalize=True).reindex(cat_tooth)

adj_means = {}
for g in cat_genus:
    preds = pred_df[pred_df['genus'] == g].set_index('tooth_class')['pred']
    adj_mean = float((preds * weights).sum())
    adj_means[g] = adj_mean

print('\nAdjusted mean num_amtl (weighted by tooth_class proportions):')
print(adj_means)

