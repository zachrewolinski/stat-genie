import pandas as pd

df = pd.read_csv('crofoot.csv')
# candidate mapping
focal_size = df['f_other']
other_size = df['win']
# candidate male/female for focal/other
focal_males = df['dist_focal']
focal_females = df['other']
other_males = df['focal']
other_females = df['f_focal']

print('Check focal size equals males+females?')
print(((focal_males + focal_females) == focal_size).value_counts())
print('Check other size equals males+females?')
print(((other_males + other_females) == other_size).value_counts())

# difference between group sizes and check unique values
rel_size = focal_size - other_size
print('Relative size summary:', rel_size.describe())

# distances
print('m_other range', df['m_other'].min(), df['m_other'].max())
print('n_focal range', df['n_focal'].min(), df['n_focal'].max())

# compute focal home advantage (1 if focal closer to its home than other to its home)
home_adv = (df['m_other'] < df['n_focal']).astype(int)
print('Home advantage counts (focal closer):', home_adv.value_counts())

# check if m_focal is outcome (0/1)
print('m_focal counts:', df['m_focal'].value_counts())
