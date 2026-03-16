import pandas as pd

_df = pd.read_csv('reading.csv')

# assume retake_trial is num_words, adjusted_running_time is time on page
calc_speed = _df['retake_trial'] / _df['adjusted_running_time'] * 60000

# compare with running_time
corr = calc_speed.corr(_df['running_time'])
print('corr calc_speed vs running_time', corr)
print('calc_speed summary', calc_speed.describe())
print('running_time summary', _df['running_time'].describe())

# also compare calc_speed with age (maybe adjusted time)
calc_speed2 = _df['retake_trial'] / _df['age'] * 60000
print('corr calc_speed2 vs running_time', calc_speed2.corr(_df['running_time']))

# check correlation with gender (scrolling time) etc
print('corr adjusted_running_time vs age', _df['adjusted_running_time'].corr(_df['age']))
