import pandas as pd

# Load data
df = pd.read_csv("boxes.csv")

# Columns
# feature1: outcome 1=undemonstrated, 2=majority, 3=minority
# feature2: gender 1=girl, 2=boy
# feature3: age (years)
# feature4: majority-first (0/1)
# feature5: site (1-8)

# Basic proportions
n = len(df)
prop_majority = (df["feature1"] == 2).mean()
prop_minority = (df["feature1"] == 3).mean()
prop_undemonstrated = (df["feature1"] == 1).mean()

# Age trend: majority choice vs age
age_majority_corr = df[["feature3", "feature1"]].assign(
    is_majority=lambda d: (d["feature1"] == 2).astype(int)
)[["feature3", "is_majority"]].corr().iloc[0, 1]

# Site-level variation in majority choice
site_majority = df.assign(is_majority=(df["feature1"] == 2).astype(int)).groupby("feature5")["is_majority"].mean()
site_range = site_majority.max() - site_majority.min()

# Very simple scalar construction:
# - Start from baseline reflecting overall majority preference.
#   Center at 0.33 (chance among 3 options) and scale.
baseline = (prop_majority - 1/3) / (2/3)  # in [-0.5, 1]

# - Add contribution from age trend (bounded to [-1,1]).
age_term = max(-1.0, min(1.0, age_majority_corr))

# - Subtract contribution from cross-cultural variability (more variation -> less universal yes).
#   site_range is in [0,1]. Map to [-0.5,0].
site_term = -0.5 * site_range

raw_score = 0.5 * baseline + 0.3 * age_term + 0.2 * site_term

# Map raw_score in approx [-1,1] to Likert -100..100
scalar = int(round(raw_score * 100))

print('SUMMARY')
print(f'n = {n}')
print(f'prop_majority = {prop_majority:.3f}, prop_minority = {prop_minority:.3f}, prop_undemonstrated = {prop_undemonstrated:.3f}')
print(f'age_majority_corr = {age_majority_corr:.3f}')
print('site_majority proportions:')
print(site_majority)
print(f'raw_score = {raw_score:.3f}, scalar = {scalar}')

with open('conclusion.txt', 'w') as f:
    f.write(str(scalar))
