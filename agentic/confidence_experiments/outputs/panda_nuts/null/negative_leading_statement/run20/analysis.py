import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv('panda_nuts.csv')

# Basic cleaning
# Ensure categorical
for col in ['sex', 'help', 'hammer', 'chimpanzee']:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Efficiency (rate per second)
df['rate'] = df['nuts_opened'] / df['seconds']

# Poisson GLM with offset for exposure time
formula = 'nuts_opened ~ age + C(sex) + C(help)'
model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df['seconds'])
)
res = model.fit(cov_type='cluster', cov_kwds={'groups': df['chimpanzee']})

# Check overdispersion (Pearson chi2 / df)
pearson = res.pearson_chi2
ratio = pearson / res.df_resid

# Negative binomial (robust cluster SE)
nb_model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.NegativeBinomial(alpha=1.0),
    offset=np.log(df['seconds'])
)
nb_res = nb_model.fit(cov_type='cluster', cov_kwds={'groups': df['chimpanzee']})

# Assemble key stats

def coef_table(res):
    params = res.params
    bse = res.bse
    pvalues = res.pvalues
    irrs = np.exp(params)
    # 95% CI for IRR
    ci_low = np.exp(params - 1.96*bse)
    ci_high = np.exp(params + 1.96*bse)
    out = pd.DataFrame({
        'coef': params,
        'se': bse,
        'p': pvalues,
        'IRR': irrs,
        'IRR_95_low': ci_low,
        'IRR_95_high': ci_high,
    })
    return out

poisson_tbl = coef_table(res)
nb_tbl = coef_table(nb_res)

# Descriptive summary
summary = {
    'n_rows': int(df.shape[0]),
    'n_chimpanzees': int(df['chimpanzee'].nunique()),
    'rate_mean': float(df['rate'].mean()),
    'rate_median': float(df['rate'].median()),
    'rate_std': float(df['rate'].std()),
    'pearson_overdispersion_ratio': float(ratio),
}

poisson_out = poisson_tbl.loc[['age', 'C(sex)[T.m]', 'C(help)[T.y]']].to_dict(orient='index')
nb_out = nb_tbl.loc[['age', 'C(sex)[T.m]', 'C(help)[T.y]']].to_dict(orient='index')

output = {
    'summary': summary,
    'poisson_cluster': poisson_out,
    'neg_bin_cluster': nb_out,
}

with open('analysis_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(json.dumps(output, indent=2))
