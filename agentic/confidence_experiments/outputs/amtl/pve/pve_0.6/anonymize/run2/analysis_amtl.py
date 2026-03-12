import pandas as pd
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('amtl.csv')

# Basic cleaning: drop rows with missing values in key fields
cols = ['feature3', 'feature5', 'feature7', 'feature1', 'feature8', 'feature4']
clean = df.dropna(subset=cols).copy()

# Indicator for Homo sapiens vs non-human primates
clean['is_human'] = (clean['feature8'] == 'Homo sapiens').astype(int)

# Fit OLS with robust SEs
ols_model = smf.ols('feature3 ~ is_human + feature5 + feature7 + C(feature1)', data=clean).fit(cov_type='HC3')

# Fit WLS using observable sockets as weights (more sockets -> more reliable)
wls_model = smf.wls('feature3 ~ is_human + feature5 + feature7 + C(feature1)', data=clean, weights=clean['feature4']).fit(cov_type='HC3')

print('N', len(clean))
print('\nOLS HC3')
print(ols_model.summary().tables[1])
print('\nWLS HC3')
print(wls_model.summary().tables[1])

# Compute adjusted mean difference for human vs non-human at mean covariates
mean_cov = {
    'feature5': clean['feature5'].mean(),
    'feature7': clean['feature7'].mean(),
    'feature1': 'Anterior'
}

# Predicted difference using OLS
pred_non = ols_model.predict(pd.DataFrame([{**mean_cov, 'is_human': 0}]))[0]
pred_hum = ols_model.predict(pd.DataFrame([{**mean_cov, 'is_human': 1}]))[0]
print('\nOLS predicted difference (human - nonhuman) at mean covariates:', pred_hum - pred_non)

pred_non_w = wls_model.predict(pd.DataFrame([{**mean_cov, 'is_human': 0}]))[0]
pred_hum_w = wls_model.predict(pd.DataFrame([{**mean_cov, 'is_human': 1}]))[0]
print('WLS predicted difference (human - nonhuman) at mean covariates:', pred_hum_w - pred_non_w)
