import pandas as pd
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('amtl.csv')

# human indicator

df['human'] = (df['tooth_class'] == 'Homo sapiens').astype(int)

# OLS with cluster robust SE by specimen
model = smf.ols('genus ~ human + pop + stdev_age + C(sockets)', data=df).fit(cov_type='cluster', cov_kwds={'groups': df['prob_male']})

coef = model.params['human']
pval = model.pvalues['human']
ci_low, ci_high = model.conf_int().loc['human']

print('human_coef', coef)
print('human_pval', pval)
print('human_ci', ci_low, ci_high)

# predicted means by genus categories from model with genus categories
model2 = smf.ols('genus ~ C(tooth_class) + pop + stdev_age + C(sockets)', data=df).fit(cov_type='cluster', cov_kwds={'groups': df['prob_male']})

# compute adjusted mean differences for genus categories
cats = sorted(df['tooth_class'].unique())
mean_pop = df['pop'].mean()
mean_sex = df['stdev_age'].mean()
ref_socket = df['sockets'].unique()[0]

preds = {}
for g in cats:
    row = pd.DataFrame({'tooth_class':[g], 'pop':[mean_pop], 'stdev_age':[mean_sex], 'sockets':[ref_socket]})
    preds[g] = float(model2.predict(row)[0])

print('preds', preds)
