import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

DATA_PATH = 'teachingratings.csv'

df = pd.read_csv(DATA_PATH)
print('Columns:', df.columns.tolist())
print('Shape:', df.shape)

# Keep only relevant columns
# beauty and eval should exist

# Basic summary
print(df[['beauty','eval']].describe())

# Pearson correlation
corr = df[['beauty','eval']].corr().iloc[0,1]
pearson = stats.pearsonr(df['beauty'], df['eval'])
print('Pearson r:', corr, pearson)

# Simple OLS
X = sm.add_constant(df['beauty'])
model = sm.OLS(df['eval'], X).fit()
print(model.summary())

# Multivariate controls (common in paper). choose available columns: age, gender, minority, native, tenure, division, credits, students, allstudents.
# Need to encode categorical.

controls = ['age','students','allstudents','minority','native','tenure','division','credits','gender']
# some may not exist
controls = [c for c in controls if c in df.columns]

# Create design matrix
X_controls = df[controls].copy()
# encode categories
cat_cols = X_controls.select_dtypes(include=['object','category']).columns
X_controls = pd.get_dummies(X_controls, columns=cat_cols, drop_first=True)

X2 = pd.concat([df['beauty'], X_controls], axis=1)
X2 = sm.add_constant(X2)
model2 = sm.OLS(df['eval'], X2).fit()
print(model2.summary())

# Standardized effect of beauty
# standardize beauty and eval for effect size
z_beauty = (df['beauty'] - df['beauty'].mean())/df['beauty'].std(ddof=0)
# OLS standardized with eval maybe not needed; compute coefficient for z-beauty
Xz = sm.add_constant(z_beauty)
model_z = sm.OLS(df['eval'], Xz).fit()
print('Std beauty coef (per 1 sd):', model_z.params['beauty'])

# Save key stats
results = {
    'n': int(df.shape[0]),
    'pearson_r': float(pearson.statistic),
    'pearson_p': float(pearson.pvalue),
    'simple_coef': float(model.params['beauty']),
    'simple_p': float(model.pvalues['beauty']),
    'simple_ci': [float(model.conf_int().loc['beauty',0]), float(model.conf_int().loc['beauty',1])],
    'adj_coef': float(model2.params['beauty']),
    'adj_p': float(model2.pvalues['beauty']),
    'adj_ci': [float(model2.conf_int().loc['beauty',0]), float(model2.conf_int().loc['beauty',1])],
    'std_beauty_effect': float(model_z.params['beauty']),
}
print('RESULTS_JSON', json.dumps(results, indent=2))
