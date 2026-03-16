import pandas as pd

_df = pd.read_csv('reading.csv')

num_cols = [c for c in _df.columns if pd.api.types.is_numeric_dtype(_df[c])]

# candidate for num_words per metadata: retake_trial
num_words = _df['retake_trial']

# candidate times: adjusted_running_time, age, gender
for time_col in ['adjusted_running_time','age','gender']:
    # avoid zero
    speed_calc = num_words / (_df[time_col] / 60000.0)
    print('\nUsing time', time_col)
    for col in num_cols:
        corr = speed_calc.corr(_df[col])
        if abs(corr) > 0.2:
            print('  corr with', col, corr)
