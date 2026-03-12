import pandas as pd
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv("amtl.csv")

df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

formula = 'num_amtl ~ is_human + age + prob_male + C(tooth_class)'
model = smf.ols(formula, data=df).fit(cov_type='HC3')

print(model.summary())

coef = model.params['is_human']
pval = model.pvalues['is_human']
print(f"\nHuman vs nonhuman coef={coef:.3f}, p={pval:.4g}")
