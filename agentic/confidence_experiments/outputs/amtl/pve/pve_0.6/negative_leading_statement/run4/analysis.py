import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
_df = pd.read_csv('amtl.csv')

# Basic cleaning: drop rows with missing covariates/outcome
cols = ['num_amtl','age','prob_male','tooth_class','genus']
df = _df.dropna(subset=cols).copy()

# Binary human indicator
_df['human'] = (_df['genus'] == 'Homo sapiens').astype(int)

# Model 1: human vs non-human, adjust for age, sex, tooth class
m1 = smf.ols('num_amtl ~ human + age + prob_male + C(tooth_class)', data=_df).fit(cov_type='HC3')

# Model 2: full genus factor
m2 = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=_df).fit(cov_type='HC3')

# Get adjusted means (least squares means) for each genus at mean covariates and tooth_class distribution
# Use sample-weighted proportions for tooth_class
mean_age = _df['age'].mean()
mean_prob_male = _df['prob_male'].mean()
# Proportions for tooth_class
class_props = _df['tooth_class'].value_counts(normalize=True)

# Build prediction dataframe for each genus with weighted tooth_class mix
pred_rows = []
for genus in sorted(_df['genus'].unique()):
    for tooth_class, w in class_props.items():
        pred_rows.append({
            'genus': genus,
            'age': mean_age,
            'prob_male': mean_prob_male,
            'tooth_class': tooth_class,
            'weight': w,
        })

pred_df = pd.DataFrame(pred_rows)
# Predict and compute weighted mean per genus
pred_df['pred'] = m2.predict(pred_df)
lsmeans = pred_df.groupby('genus').apply(lambda g: np.average(g['pred'], weights=g['weight']))

# Pairwise contrasts: Homo sapiens vs each other genus
# Build contrast vectors for m2 parameters
params = m2.params
# Parameter names: Intercept, C(genus)[T.Pan], etc; reference is alphabetically first? statsmodels default sorts.
# We'll identify and compute contrasts using t_test with appropriate vector.
param_names = list(params.index)

# Helper to build contrast vector comparing Homo sapiens to another genus
# We use reference genus as baseline (the one omitted in params)
# We compute predicted mean at reference covariates, but differences between genera only depend on genus terms.

# Identify reference genus
# statsmodels sets reference as first alphabetically for C(genus) unless specified.
# We'll infer it as the genus without a parameter term.
all_genus = sorted(_df['genus'].unique())
# Terms present for genus
present = [name for name in param_names if name.startswith('C(genus)')]
# Extract genus names from terms like C(genus)[T.Pan]
term_genus = [name.split('T.')[-1].rstrip(']') for name in present]
# Reference genus is the one not in term_genus
ref_genus = [g for g in all_genus if g not in term_genus]
ref_genus = ref_genus[0] if ref_genus else None

# Function to get contrast vector for genus difference g1 - g2
# With treatment coding, mean for genus = Intercept + term (if not reference)
# Difference between g1 and g2 uses their term coefficients; intercept cancels.
def contrast_vector(g1, g2):
    v = np.zeros(len(param_names))
    def term_index(genus):
        term = f'C(genus)[T.{genus}]'
        return param_names.index(term) if term in param_names else None
    idx1 = term_index(g1)
    idx2 = term_index(g2)
    if idx1 is not None:
        v[idx1] = 1.0
    if idx2 is not None:
        v[idx2] -= 1.0
    return v

# Compute pairwise tests Homo sapiens vs others
pairwise = {}
for g in all_genus:
    if g == 'Homo sapiens':
        continue
    v = contrast_vector('Homo sapiens', g)
    ttest = m2.t_test(v)
    pairwise[g] = {
        'diff': float(ttest.effect),
        't': float(ttest.tvalue),
        'p': float(ttest.pvalue)
    }

# Also compute human vs non-human effect from model 1
human_effect = {
    'coef': float(m1.params['human']),
    't': float(m1.tvalues['human']),
    'p': float(m1.pvalues['human'])
}

# Compute adjusted mean for non-human average (weighted by their frequency)
nonhuman = _df[_df['genus'] != 'Homo sapiens']
# Use same lsmeans weighting by sample sizes for non-human genera
nonhuman_weights = nonhuman['genus'].value_counts(normalize=True)
nonhuman_mean = float(sum(lsmeans[g] * nonhuman_weights[g] for g in nonhuman_weights.index))

results = {
    'n': int(len(_df)),
    'ref_genus': ref_genus,
    'human_vs_nonhuman': human_effect,
    'lsmeans': lsmeans.to_dict(),
    'nonhuman_lsmean': nonhuman_mean,
    'pairwise': pairwise,
}

print('RESULTS')
for k, v in results.items():
    print(k, v)
