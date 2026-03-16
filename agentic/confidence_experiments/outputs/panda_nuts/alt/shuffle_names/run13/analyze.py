import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

pd.set_option('display.max_columns', None)

df = pd.read_csv('panda_nuts.csv')
print('head')
print(df.head())
print('\ninfo')
print(df.dtypes)

print('\nunique values:')
for col in df.columns:
    print('\n', col)
    print('nunique', df[col].nunique())
    if df[col].dtype == object:
        print('unique', sorted(df[col].unique())[:20])
    else:
        print('min', df[col].min(), 'max', df[col].max())
        print('mean', df[col].mean())

# try to infer which column is what

# let's compute nuts opened per second if possible
# We'll test all numeric columns as possible duration and count
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
print('\nnumeric cols', num_cols)

# pairwise show distributions for numeric columns
summary = df[num_cols].describe().T
print('\nsummary numeric')
print(summary)

# help column appears numeric; seconds maybe categorical y/N. We'll map help indicator.

# A likely mapping: seconds column is help indicator, nuts_opened is sex, sex is hammer type.

# We'll prepare a candidate dataframe with inferred columns
# Identify sex column: contains 'm' and 'f'
sex_col = None
for c in df.columns:
    if df[c].dtype == object and set(df[c].unique()) <= set(['m','f']):
        sex_col = c
print('sex_col', sex_col)

# help indicator column: contains y/N (case?)
help_col = None
for c in df.columns:
    if df[c].dtype == object and set(df[c].unique()) <= set(['y','N','n','Y']):
        help_col = c
print('help_col', help_col)

# hammer type column: categorical with 4 types maybe wood/Q/G/?
# find object column with 4 unique values
hammer_col = None
for c in df.columns:
    if df[c].dtype == object and df[c].nunique()==4:
        hammer_col = c
print('hammer_col', hammer_col)

# guess: nuts_opened count column should be integer count maybe 0-77, likely numeric with many unique and non-integer?
# session seconds maybe numeric with decimal .0 maybe 2.5-135
# age maybe integer 1-22

print('\npossible age column:')
for c in num_cols:
    if df[c].min()>=0 and df[c].max()<=30:
        print(c, df[c].min(), df[c].max(), df[c].nunique())

print('\npossible duration column:')
for c in num_cols:
    if df[c].max()>=60 and df[c].max()<=200:
        print(c, df[c].min(), df[c].max(), df[c].nunique())

print('\npossible nuts_opened count:')
for c in num_cols:
    if df[c].max()>=20 and df[c].max()<=100:
        print(c, df[c].min(), df[c].max(), df[c].nunique())

# Let's interpret based on data
# We'll create final variables for analysis.

age = df['age']
# but check if age seems integer 1-22? yes maybe.

# We'll infer mapping by distribution check

