import json
import pandas as pd
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('amtl.csv')

# Clean / encode

df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# OLS model with pooled non-human comparison
model = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class)', data=df).fit()

# Genus-specific model for pairwise comparisons
model_genus = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=df).fit()

# Extract coefficient for is_human
coef_is_human = model.params['is_human']
se_is_human = model.bse['is_human']
p_is_human = model.pvalues['is_human']

# Pairwise contrasts: Homo sapiens vs each other genus
# With reference category chosen by statsmodels (alphabetical by default), compute contrasts.
# We will compute predicted difference between Homo sapiens and each other genus by re-leveling.

pairwise = {}
for other in ['Pan', 'Papio', 'Pongo']:
    # Create a temp categorical with specified order to set reference
    tmp = df.copy()
    # Ensure all categories present
    cats = ['Homo sapiens', other]
    # Recode to only two levels for direct comparison
    tmp = tmp[tmp['genus'].isin(cats)].copy()
    tmp['genus'] = pd.Categorical(tmp['genus'], categories=[other, 'Homo sapiens'])
    m = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=tmp).fit()
    # Coefficient for Homo sapiens relative to other
    coef = m.params.get('C(genus)[T.Homo sapiens]', float('nan'))
    pval = m.pvalues.get('C(genus)[T.Homo sapiens]', float('nan'))
    pairwise[other] = {'coef': coef, 'pval': pval, 'n': int(tmp.shape[0])}

# Summaries for output
summary = {
    'n': int(df.shape[0]),
    'coef_is_human': float(coef_is_human),
    'se_is_human': float(se_is_human),
    'p_is_human': float(p_is_human),
    'pairwise': pairwise,
    'model_r2': float(model.rsquared),
    'model_adj_r2': float(model.rsquared_adj),
}

print(json.dumps(summary, indent=2))
