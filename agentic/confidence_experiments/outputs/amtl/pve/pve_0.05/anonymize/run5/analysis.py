import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data
df = pd.read_csv('amtl.csv')

# Basic cleaning
# Create human indicator
df['is_human'] = (df['feature8'] == 'Homo sapiens').astype(int)

# OLS regression with robust SE
model = smf.ols('feature3 ~ is_human + feature5 + feature7 + C(feature1)', data=df).fit(cov_type='HC3')

# Extract coefficient for is_human
coef = model.params['is_human']
se = model.bse['is_human']
pval = model.pvalues['is_human']

# Adjusted means using model: predict at mean age/sex and each tooth class? We'll compute average marginal effect of human vs nonhuman
# Compute predicted values for each row setting is_human=1 and 0, then average.
df_h = df.copy()
df_h['is_human'] = 1
df_n = df.copy()
df_n['is_human'] = 0
pred_h = model.predict(df_h).mean()
pred_n = model.predict(df_n).mean()
diff = pred_h - pred_n

# Save summary to file for inspection
with open('analysis_summary.txt', 'w') as f:
    f.write(model.summary().as_text())
    f.write('\n\n')
    f.write(f"coef_is_human={coef}\nse={se}\npval={pval}\n")
    f.write(f"pred_h_mean={pred_h}\npred_n_mean={pred_n}\ndiff={diff}\n")

print('coef_is_human', coef)
print('se', se)
print('pval', pval)
print('pred_h_mean', pred_h)
print('pred_n_mean', pred_n)
print('diff', diff)
