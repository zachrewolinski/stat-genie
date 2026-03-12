import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

raw = pd.read_csv('amtl.csv')

# Map columns to meaning based on info.json descriptions
# tooth_class: sockets column
# genus: tooth_class column
# missing teeth count: genus column (noised)
# total sockets: age column
# age at death: pop column
# sex probability: stdev_age column

df = pd.DataFrame({
    'tooth_class': raw['sockets'],
    'genus': raw['tooth_class'],
    'missing_raw': raw['genus'],
    'total_raw': raw['age'],
    'age_at_death': raw['pop'],
    'sex_prob': raw['stdev_age'],
})

# Keep positive totals
df = df[df['total_raw'] > 0].copy()

# Clip missing counts to [0, total]
missing_clipped = df['missing_raw'].clip(lower=0)
missing_clipped = np.minimum(missing_clipped, df['total_raw'])

# Response proportion and weights
df['missing'] = missing_clipped
df['prop'] = df['missing'] / df['total_raw']

# Set categorical baselines
df['genus'] = pd.Categorical(df['genus'], categories=['Homo sapiens', 'Pan', 'Pongo', 'Papio'])
df['tooth_class'] = pd.Categorical(df['tooth_class'], categories=['Anterior', 'Posterior', 'Premolar'])

# Fit binomial GLM (proportion with weights)
model = smf.glm(
    formula='prop ~ C(genus) + age_at_death + sex_prob + C(tooth_class)',
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df['total_raw']
).fit()

print(model.summary())

# Extract genus coefficients vs Homo baseline
coef = model.params
pvals = model.pvalues

for g in ['C(genus)[T.Pan]', 'C(genus)[T.Pongo]', 'C(genus)[T.Papio]']:
    if g in coef:
        print(g, 'coef', float(coef[g]), 'OR', float(np.exp(coef[g])), 'p', float(pvals[g]))

# Predicted mean proportions for Homo vs others at mean covariates
mean_age = df['age_at_death'].mean()
mean_sex = df['sex_prob'].mean()

def pred_for(genus):
    preds = []
    for tc in df['tooth_class'].cat.categories:
        row = pd.DataFrame({
            'genus':[genus],
            'age_at_death':[mean_age],
            'sex_prob':[mean_sex],
            'tooth_class':[tc]
        })
        preds.append(model.predict(row)[0])
    return float(np.mean(preds))

for genus in ['Homo sapiens','Pan','Pongo','Papio']:
    print('Pred', genus, pred_for(genus))

# Likelihood ratio test for genus
reduced = smf.glm(
    formula='prop ~ age_at_death + sex_prob + C(tooth_class)',
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df['total_raw']
).fit()

lr_stat = 2*(model.llf - reduced.llf)
p_lr = stats.chi2.sf(lr_stat, df=3)
print('LR test genus chi2', float(lr_stat), 'p', float(p_lr))
