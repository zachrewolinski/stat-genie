import pandas as pd


df = pd.read_csv('amtl.csv')

# check for each specimen ID (prob_male) if num_amtl/pog/pop constant
spec_groups = df.groupby('prob_male')

same_num_amtl = (spec_groups['num_amtl'].nunique() == 1).mean()
same_pop = (spec_groups['pop'].nunique() == 1).mean()
same_age = (spec_groups['age'].nunique() == 1).mean()

print('share specimens with constant num_amtl', same_num_amtl)
print('share specimens with constant pop', same_pop)
print('share specimens with constant age', same_age)

# check if age varies by tooth class within specimen
var_age = (spec_groups['age'].nunique() > 1).mean()
print('share specimens with varying age', var_age)

# check if genus (numeric) varies within specimen
same_genus = (spec_groups['genus'].nunique() == 1).mean()
print('share specimens with constant genus', same_genus)

# check counts per specimen
print('rows per specimen counts')
print(spec_groups.size().value_counts().head())

