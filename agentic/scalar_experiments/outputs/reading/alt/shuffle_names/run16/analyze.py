import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 200)

df = pd.read_csv('reading.csv')

summary = []
for col in df.columns:
    series = df[col]
    info = {
        'col': col,
        'dtype': series.dtype,
        'n_unique': series.nunique(dropna=True),
        'n_missing': series.isna().sum(),
    }
    if pd.api.types.is_numeric_dtype(series):
        info.update({
            'min': series.min(),
            'max': series.max(),
            'mean': series.mean(),
            'std': series.std(),
        })
        # include a few samples
        info['samples'] = series.dropna().unique()[:5]
    else:
        info['samples'] = series.dropna().unique()[:5]
    summary.append(info)

summary_df = pd.DataFrame(summary)
print(summary_df)

# Also check correlation with num_words or running_time if speed is calculated

