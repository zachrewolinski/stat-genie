import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Rename for clarity
_df = _df.rename(columns={
    'feature1': 'tooth_class',
    'feature3': 'amtl',
    'feature4': 'sockets',
    'feature5': 'age',
    'feature6': 'age_uncertainty',
    'feature7': 'sex',
    'feature8': 'genus',
})

# Ensure categorical types
_df['tooth_class'] = _df['tooth_class'].astype('category')
_df['genus'] = _df['genus'].astype('category')

# Primary model: adjusted for age, sex, tooth class
model = smf.ols(
    "amtl ~ C(genus, Treatment(reference='Homo sapiens')) + C(tooth_class) + age + sex",
    data=_df,
).fit(cov_type='HC3')

params = model.params

# Average difference of non-human genera vs Homo (negative means non-human lower)
coef_names = [
    "C(genus, Treatment(reference='Homo sapiens'))[T.Pan]",
    "C(genus, Treatment(reference='Homo sapiens'))[T.Papio]",
    "C(genus, Treatment(reference='Homo sapiens'))[T.Pongo]",
]

avg_coef = np.mean([params.get(name, np.nan) for name in coef_names])

# t-test for average difference vs 0
# Build contrast vector
exog_names = model.model.exog_names
contrast = np.zeros(len(exog_names))
for name in coef_names:
    if name in exog_names:
        contrast[exog_names.index(name)] = 1/3

avg_test = model.t_test(contrast)

# Compute adjusted mean difference (Homo - nonhuman average)
# Homo baseline = 0, nonhuman avg = avg_coef
homo_minus_nonhuman = -avg_coef

# Residual SD
resid_sd = np.sqrt(model.scale)

# Sensitivity model adding sockets (observable sockets)
model_sockets = smf.ols(
    "amtl ~ C(genus, Treatment(reference='Homo sapiens')) + C(tooth_class) + age + sex + sockets",
    data=_df,
).fit(cov_type='HC3')

params2 = model_sockets.params
avg_coef2 = np.mean([params2.get(name, np.nan) for name in coef_names])
exog_names2 = model_sockets.model.exog_names
contrast2 = np.zeros(len(exog_names2))
for name in coef_names:
    if name in exog_names2:
        contrast2[exog_names2.index(name)] = 1/3
avg_test2 = model_sockets.t_test(contrast2)

# Package results
result = {
    'n': int(len(_df)),
    'coef_pan': float(params.get(coef_names[0], np.nan)),
    'coef_papio': float(params.get(coef_names[1], np.nan)),
    'coef_pongo': float(params.get(coef_names[2], np.nan)),
    'avg_nonhuman_vs_homo_coef': float(avg_coef),
    'homo_minus_nonhuman': float(homo_minus_nonhuman),
    'avg_diff_pvalue': float(avg_test.pvalue),
    'resid_sd': float(resid_sd),
    'model_r2': float(model.rsquared),
    'sockets_avg_coef': float(avg_coef2),
    'sockets_avg_pvalue': float(avg_test2.pvalue),
}

print(result)
