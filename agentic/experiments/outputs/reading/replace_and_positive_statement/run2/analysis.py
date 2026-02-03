import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import ttest_ind
import statsmodels.formula.api as smf

# Load data
DATA_PATH = 'reading.csv'
df = pd.read_csv(DATA_PATH)

# Focus on individuals with dyslexia
# Some columns appear noisy; round to nearest label to recover intended groups.
if 'dyslexia_bin' in df.columns:
    dys_flag = df['dyslexia_bin'].round().clip(0, 1)
    dys_df = df[dys_flag == 1].copy()
elif 'dyslexia' in df.columns:
    dys_flag = df['dyslexia'].round().clip(0, 2)
    dys_df = df[dys_flag >= 1].copy()
else:
    dys_df = df.copy()

# Basic cleaning: keep positive speeds
if 'speed' in dys_df.columns:
    dys_df = dys_df[np.isfinite(dys_df['speed'])].copy()
    dys_df = dys_df[dys_df['speed'] > 0].copy()

# Group comparison: reader_view on/off
rv1 = dys_df[dys_df['reader_view'] == 1]['speed']
rv0 = dys_df[dys_df['reader_view'] == 0]['speed']

mean1 = rv1.mean()
mean0 = rv0.mean()
median1 = rv1.median()
median0 = rv0.median()

# Welch t-test
if len(rv1) > 1 and len(rv0) > 1:
    t_stat, p_val, dfree = ttest_ind(rv1, rv0, usevar='unequal')
else:
    t_stat, p_val, dfree = np.nan, np.nan, np.nan

# Effect size (Cohen's d)
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    return (x.mean() - y.mean()) / np.sqrt(pooled) if pooled > 0 else np.nan

d_val = cohens_d(rv1, rv0)

# Regression with controls (log speed)
dys_df = dys_df.copy()
dys_df['log_speed'] = np.log(dys_df['speed'])

# Build formula with available columns
controls = []
for col in ['num_words', 'Flesch_Kincaid', 'age', 'img_width', 'retake_trial']:
    if col in dys_df.columns:
        controls.append(col)

# Categorical controls (only for object/category dtypes and >1 level)
cat_cols = ['page_id', 'device', 'education', 'gender', 'language', 'english_native']
for col in cat_cols:
    if col in dys_df.columns and dys_df[col].nunique(dropna=True) > 1:
        if pd.api.types.is_object_dtype(dys_df[col]) or pd.api.types.is_categorical_dtype(dys_df[col]):
            controls.append(f'C({col})')

formula = 'log_speed ~ reader_view'
if controls:
    formula += ' + ' + ' + '.join(controls)

# Drop rows with missing values for model variables
model_cols = ['log_speed', 'reader_view']
for col in controls:
    # strip C() to get raw column name
    if col.startswith('C(') and col.endswith(')'):
        model_cols.append(col[2:-1])
    else:
        model_cols.append(col)
model_df = dys_df[model_cols].dropna()

model = None
if len(model_df) > 5 and model_df['reader_view'].nunique() > 1:
    model = smf.ols(formula, data=model_df).fit()

print('Dyslexia subset size:', len(dys_df))
print('Reader view ON count:', len(rv1))
print('Reader view OFF count:', len(rv0))
print('Speed mean (ON):', mean1)
print('Speed mean (OFF):', mean0)
print('Speed median (ON):', median1)
print('Speed median (OFF):', median0)
print('Welch t-test: t=%.3f, p=%.6f, df=%.1f' % (t_stat, p_val, dfree))
print("Cohen's d (ON-OFF):", d_val)
print('\nRegression summary (log_speed):')
if model is not None:
    print(model.summary())
else:
    print('Not enough data for regression after filtering/dropping missing values.')
