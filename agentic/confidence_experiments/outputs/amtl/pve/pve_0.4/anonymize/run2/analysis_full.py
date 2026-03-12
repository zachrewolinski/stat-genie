import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

path = 'amtl.csv'
df = pd.read_csv(path)
for col in ['feature1', 'feature8']:
    df[col] = df[col].astype('category')

formula = "feature3 ~ C(feature8, Treatment(reference='Homo sapiens')) + C(feature1) + feature5 + feature7"
model = smf.ols(formula, data=df).fit(cov_type='HC3')

coef_table = model.summary2().tables[1]
coef_genus = coef_table.loc[[
    "C(feature8, Treatment(reference='Homo sapiens'))[T.Pan]",
    "C(feature8, Treatment(reference='Homo sapiens'))[T.Papio]",
    "C(feature8, Treatment(reference='Homo sapiens'))[T.Pongo]",
]]

means = {}
for genus in df['feature8'].cat.categories:
    df_tmp = df.copy()
    df_tmp['feature8'] = genus
    means[genus] = model.predict(df_tmp).mean()

print('Model formula:', formula)
print('\nGenus coefficients (relative to Homo sapiens):')
print(coef_genus.to_string())
print('\nCovariate-standardized predicted means by genus:')
for k, v in means.items():
    print(f"{k}: {v:.4f}")

homo_mean = means['Homo sapiens']
print('\nAdjusted mean differences vs Homo sapiens:')
for genus, mean_val in means.items():
    if genus == 'Homo sapiens':
        continue
    print(f"{genus}: {mean_val - homo_mean:.4f}")

# Print overall model R2 and N
print(f"\nN={int(model.nobs)}, R2={model.rsquared:.3f}")
