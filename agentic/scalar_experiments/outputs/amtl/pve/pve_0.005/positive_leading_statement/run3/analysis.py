import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
from patsy import build_design_matrices

# Load data
df = pd.read_csv('amtl.csv')

# Ensure categorical ordering

genus_order = ['Homo sapiens', 'Pan', 'Pongo', 'Papio']
tooth_order = ['Anterior', 'Premolar', 'Posterior']

df['genus'] = pd.Categorical(df['genus'], categories=genus_order, ordered=False)
df['tooth_class'] = pd.Categorical(df['tooth_class'], categories=tooth_order, ordered=False)

# Fit linear model with robust SEs
model = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')

# Build adjusted mean predictions for each genus using average over covariate distribution

def adjusted_mean_and_se(genus_value):
    df_copy = df.copy()
    df_copy['genus'] = genus_value
    # Build design matrix for new data
    design_info = model.model.data.design_info
    exog = build_design_matrices([design_info], df_copy)[0]
    # Average row vector
    xbar = exog.mean(axis=0)
    mean_pred = float(np.dot(xbar, model.params))
    cov = model.cov_params()
    se = float(np.sqrt(np.dot(xbar, np.dot(cov, xbar))))
    return mean_pred, se, xbar

adjusted = {}
for g in genus_order:
    mean_pred, se, xbar = adjusted_mean_and_se(g)
    adjusted[g] = {'mean': mean_pred, 'se': se, 'xbar': xbar}

# Pairwise contrasts: Homo sapiens vs each nonhuman genus
results = []
for g in ['Pan', 'Pongo', 'Papio']:
    contrast = adjusted['Homo sapiens']['xbar'] - adjusted[g]['xbar']
    diff = float(np.dot(contrast, model.params))
    cov = model.cov_params()
    se_diff = float(np.sqrt(np.dot(contrast, np.dot(cov, contrast))))
    t_stat = diff / se_diff if se_diff > 0 else np.nan
    df_resid = model.df_resid
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_resid))
    results.append({
        'comparison': f'Homo sapiens - {g}',
        'diff': diff,
        'se_diff': se_diff,
        't': t_stat,
        'p': p_value
    })

# Contrast: Homo sapiens vs average of nonhuman genera
nonhuman_xbar = (adjusted['Pan']['xbar'] + adjusted['Pongo']['xbar'] + adjusted['Papio']['xbar']) / 3
contrast = adjusted['Homo sapiens']['xbar'] - nonhuman_xbar
diff = float(np.dot(contrast, model.params))
cov = model.cov_params()
se_diff = float(np.sqrt(np.dot(contrast, np.dot(cov, contrast))))
t_stat = diff / se_diff if se_diff > 0 else np.nan
df_resid = model.df_resid
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_resid))

summary = {
    'adjusted_means': {g: adjusted[g]['mean'] for g in genus_order},
    'pairwise': results,
    'homo_vs_nonhuman_avg': {
        'diff': diff,
        'se_diff': se_diff,
        't': t_stat,
        'p': p_value
    },
    'model_summary': {
        'n': int(model.nobs),
        'df_resid': float(model.df_resid),
        'r2': float(model.rsquared)
    }
}

print(summary)
