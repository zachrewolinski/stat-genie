import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
df = pd.read_csv('amtl.csv')

# Ensure categorical types
df['genus'] = df['genus'].astype('category')
df['tooth_class'] = df['tooth_class'].astype('category')

# Fit linear model (Gaussian) controlling for age, sex, tooth class
formula = 'num_amtl ~ C(genus, Treatment(reference="Homo sapiens")) + age + prob_male + C(tooth_class)'
model = smf.ols(formula, data=df).fit()

# Extract coefficients for non-human genera
params = model.params
cov = model.cov_params()

# Contrast: Homo - average(non-human) = -(pan + pongo + papio)/3
contrast = np.zeros(len(params))
param_names = list(params.index)
for name, weight in [
    ('C(genus, Treatment(reference="Homo sapiens"))[T.Pan]', -1/3),
    ('C(genus, Treatment(reference="Homo sapiens"))[T.Pongo]', -1/3),
    ('C(genus, Treatment(reference="Homo sapiens"))[T.Papio]', -1/3),
]:
    if name in param_names:
        contrast[param_names.index(name)] = weight

contrast_est = float(np.dot(contrast, params))
contrast_se = float(np.sqrt(np.dot(contrast, np.dot(cov, contrast))))
t_stat = contrast_est / contrast_se if contrast_se > 0 else np.nan
p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=model.df_resid))

# Compute adjusted means at mean age/prob_male, averaged across tooth_class (equal weights)
mean_age = df['age'].mean()
mean_prob_male = df['prob_male'].mean()
tooth_levels = list(df['tooth_class'].cat.categories)


def predict_mean(genus):
    rows = []
    for tc in tooth_levels:
        rows.append({'genus': genus, 'age': mean_age, 'prob_male': mean_prob_male, 'tooth_class': tc})
    pred = model.predict(pd.DataFrame(rows))
    return float(np.mean(pred))


genus_levels = list(df['genus'].cat.categories)
adj_means = {g: predict_mean(g) for g in genus_levels}

# Effect size as standardized difference (Homo - avg non-human) / residual sd
resid_sd = float(np.sqrt(model.mse_resid))
avg_nonhuman = np.mean([adj_means[g] for g in genus_levels if g != 'Homo sapiens'])
effect = (adj_means['Homo sapiens'] - avg_nonhuman) / resid_sd if resid_sd > 0 else np.nan

# Save summary stats to json for downstream
summary = {
    'n': int(model.nobs),
    'contrast_est': contrast_est,
    'contrast_se': contrast_se,
    't_stat': t_stat,
    'p_value': p_value,
    'adj_means': adj_means,
    'resid_sd': resid_sd,
    'effect_std': effect,
}

with open('analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
