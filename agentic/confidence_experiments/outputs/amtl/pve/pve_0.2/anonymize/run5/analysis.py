import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data

df = pd.read_csv('amtl.csv')

# Ensure categorical

df['feature8'] = df['feature8'].astype('category')
df['feature1'] = df['feature1'].astype('category')

# Fit OLS model (feature3 is continuous in this dataset)
formula = "feature3 ~ C(feature8, Treatment(reference='Homo sapiens')) + feature5 + feature7 + C(feature1)"
model = smf.ols(formula, data=df).fit(cov_type='HC3')

# Extract genus coefficients (others vs Homo)
coef_table = model.summary2().tables[1]
terms = [t for t in coef_table.index if t.startswith('C(feature8')]

# Adjusted (marginal) means for each genus by averaging predictions over covariates

genera = df['feature8'].cat.categories.tolist()
mean_preds = {}
for g in genera:
    tmp = df.copy()
    tmp['feature8'] = g
    mean_preds[g] = model.predict(tmp).mean()

# Pairwise differences: Homo - other (positive means Homo higher)

diffs = {g: mean_preds['Homo sapiens'] - mean_preds[g] for g in genera if g != 'Homo sapiens'}

print('Model formula:', formula)
print('\nGenus coefficients (other vs Homo):')
print(coef_table.loc[terms][['Coef.', 'Std.Err.', 'P>|z|']])

print('\nAdjusted mean predictions by genus (marginal over covariates):')
for g, v in mean_preds.items():
    print(f'  {g}: {v:.3f}')

print('\nHomo minus other differences (positive means Homo higher):')
for g, d in diffs.items():
    print(f'  Homo - {g}: {d:.3f}')

# Overall test for genus effect (Type II ANOVA)
anova = sm.stats.anova_lm(model, typ=2)
print('\nANOVA (Type II):')
print(anova)

