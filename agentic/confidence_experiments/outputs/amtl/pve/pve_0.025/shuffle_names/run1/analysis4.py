import pandas as pd

df = pd.read_csv('amtl.csv')

# Check variability of pop and num_amtl within specimen and prob_male
for col in ['pop','num_amtl','genus','age']:
    print('\n', col)
    print('unique per specimen (mean):', df.groupby('specimen')[col].nunique().mean())
    print('unique per prob_male (mean):', df.groupby('prob_male')[col].nunique().mean())

# Check if pop or num_amtl are constant per prob_male (specimen id)
for col in ['pop','num_amtl','genus','age','stdev_age']:
    const = (df.groupby('prob_male')[col].nunique() == 1).mean()
    print(col, 'fraction constant per prob_male', const)

