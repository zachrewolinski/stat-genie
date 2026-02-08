import pandas as pd
import numpy as np
import math
from scipy import stats

# Load data
_df = pd.read_csv('affairs.csv')

# Mapping inferred from value patterns
# children yes/no is column 'religiousness'
# affairs frequency counts are column 'age' (values 0,1,2,3,7,12)

affairs = _df['age']
children = _df['religiousness']  # yes/no

# Create binary for any affair
any_affair = (affairs > 0).astype(int)

summary = _df.groupby(children).agg(
    n=('age','size'),
    mean_affairs=('age','mean'),
    median_affairs=('age','median'),
    prop_any=('age', lambda x: (x>0).mean())
)

# Difference in means and proportions
means = summary['mean_affairs']
props = summary['prop_any']

# Ensure order yes/no
mean_yes = means.get('yes', float('nan'))
mean_no = means.get('no', float('nan'))
prop_yes = props.get('yes', float('nan'))
prop_no = props.get('no', float('nan'))

# t-test for affairs counts (non-normal but ok for rough)
# use Welch t-test
x_yes = affairs[children=='yes']
x_no = affairs[children=='no']

t_stat, t_p = stats.ttest_ind(x_yes, x_no, equal_var=False)

# Mann-Whitney U test
u_stat, u_p = stats.mannwhitneyu(x_yes, x_no, alternative='two-sided')

# difference in proportions z-test
count_yes = any_affair[children=='yes'].sum()
count_no = any_affair[children=='no'].sum()
n_yes = (children=='yes').sum()
n_no = (children=='no').sum()

p_pool = (count_yes + count_no) / (n_yes + n_no)
se = math.sqrt(p_pool * (1 - p_pool) * (1/n_yes + 1/n_no))
if se == 0:
    z = float('nan')
    z_p = float('nan')
else:
    z = (count_yes/n_yes - count_no/n_no) / se
    z_p = 2 * (1 - stats.norm.cdf(abs(z)))

print('Summary by children (yes/no):')
print(summary)
print('\nMean affairs difference (no - yes):', mean_no - mean_yes)
print('Prop any difference (no - yes):', prop_no - prop_yes)
print('\nWelch t-test p:', t_p)
print('Mann-Whitney p:', u_p)
print('Prop z-test p:', z_p)
