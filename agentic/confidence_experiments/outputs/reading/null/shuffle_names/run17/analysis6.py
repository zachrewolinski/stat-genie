import pandas as pd
from scipy import stats


df = pd.read_csv('reading.csv')

# use correct_rate as dyslexia binary (1 = dyslexia), reader_view is language
sub = df[(df['correct_rate'] == 1) & df['language'].notna() & df['running_time'].notna()].copy()
rv0 = sub[sub['language'] == 0]['running_time']
rv1 = sub[sub['language'] == 1]['running_time']
print('dyslexic rows (correct_rate==1)', sub.shape[0])
print('rv0 mean', rv0.mean(), 'n', rv0.shape[0])
print('rv1 mean', rv1.mean(), 'n', rv1.shape[0])
if rv0.shape[0] > 1 and rv1.shape[0] > 1:
    t_stat, p_val = stats.ttest_ind(rv1, rv0, equal_var=False)
    print('Welch t-test t', t_stat, 'p', p_val)
