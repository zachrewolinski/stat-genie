import pandas as pd

# Load data

df = pd.read_csv('amtl.csv')
print('shape', df.shape)
print(df.head())
print('\nDtypes:')
print(df.dtypes)

# Show unique values counts for categoricals/object and summaries for numeric
for col in df.columns:
    if df[col].dtype == 'object':
        uniques = df[col].unique()
        print(f"\n{col}: {len(uniques)} uniques, sample {uniques[:10]}")
    else:
        print(f"\n{col}: min={df[col].min()} max={df[col].max()} mean={df[col].mean():.3f} std={df[col].std():.3f}")

# show correlations among numeric
num_cols = df.select_dtypes(include='number').columns
print('\nNumeric columns:', list(num_cols))
print(df[num_cols].corr())
