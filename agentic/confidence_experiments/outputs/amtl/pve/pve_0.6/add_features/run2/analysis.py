import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
df = pd.read_csv('amtl.csv')

# Basic cleaning: keep relevant columns
cols = ['num_amtl', 'age', 'prob_male', 'tooth_class', 'genus']
for c in cols:
    if c not in df.columns:
        raise ValueError(f"Missing column {c}")

# Drop rows with missing values in relevant columns
sub = df[cols].dropna()

# Create human indicator
sub['human'] = (sub['genus'] == 'Homo sapiens').astype(int)

# Fit OLS model with human indicator
model_human = smf.ols('num_amtl ~ human + age + prob_male + C(tooth_class)', data=sub).fit(cov_type='HC3')

# Fit OLS model with genus categorical to compare Homo vs each genus
model_genus = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=sub).fit(cov_type='HC3')

# Build contrasts for Homo sapiens vs each other genus
# Determine categories and baseline
cat = pd.Categorical(sub['genus'])
levels = list(cat.categories)

# Statsmodels uses first level alphabetically as baseline by default in formula
# We'll compute contrasts using t_test with the design matrix names
params = model_genus.params.index.tolist()

# Helper to build contrast vector for Homo vs other in model with categorical
# model formula includes Intercept, C(genus)[T.X] for non-baseline levels.

def contrast_homo_vs(genus_other):
    # predicted mean difference between Homo and other genus
    # Represented as linear combination of params
    # Let baseline = levels[0]
    baseline = levels[0]
    def mean_for(genus):
        # mean contribution from genus part
        if genus == baseline:
            return {'Intercept': 1.0}
        else:
            return {'Intercept': 1.0, f'C(genus)[T.{genus}]': 1.0}
    mean_h = mean_for('Homo sapiens')
    mean_o = mean_for(genus_other)
    # difference Homo - other
    contrast = {k: mean_h.get(k,0.0) - mean_o.get(k,0.0) for k in params}
    # fill missing with 0
    vec = [contrast.get(p, 0.0) for p in params]
    return vec

# Compute contrasts vs other genera
other_genera = [g for g in levels if g != 'Homo sapiens']
contrast_results = {}
for g in other_genera:
    vec = contrast_homo_vs(g)
    t = model_genus.t_test(vec)
    contrast_results[g] = {
        'diff': float(t.effect),
        't': float(t.tvalue),
        'p': float(t.pvalue),
        'se': float(t.sd)
    }

# Save key outputs
out = {
    'n_rows': int(len(df)),
    'n_used': int(len(sub)),
    'human_coef': float(model_human.params['human']),
    'human_se': float(model_human.bse['human']),
    'human_p': float(model_human.pvalues['human']),
    'human_ci_low': float(model_human.conf_int().loc['human', 0]),
    'human_ci_high': float(model_human.conf_int().loc['human', 1]),
    'contrast_results': contrast_results,
}

import json
with open('analysis_out.json', 'w') as f:
    json.dump(out, f, indent=2)

# Also output brief text summary
with open('analysis_out.txt', 'w') as f:
    f.write(model_human.summary().as_text())
    f.write("\n\n")
    f.write(model_genus.summary().as_text())
