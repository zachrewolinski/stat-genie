import pandas as pd

path='reading.csv'
df=pd.read_csv(path)

# For each participant (speed), count unique language values
uniq_lang = df.groupby('speed')['language'].nunique()
print('language unique per participant counts:')
print(uniq_lang.value_counts())

# For each participant, count unique device values
uniq_device = df.groupby('speed')['device'].nunique()
print('device unique per participant counts:')
print(uniq_device.value_counts())

# For each participant, count unique correct_rate
uniq_corr = df.groupby('speed')['correct_rate'].nunique()
print('correct_rate unique per participant counts:')
print(uniq_corr.value_counts())

# For each participant, count unique dyslexia_bin
uniq_dys_bin = df.groupby('speed')['dyslexia_bin'].nunique()
print('dyslexia_bin unique per participant counts:')
print(uniq_dys_bin.value_counts())
