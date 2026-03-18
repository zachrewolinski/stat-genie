import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('affairs.csv')

candidates = ['education','age']

# actual children variable (binary) likely in 'religiousness' column (yes/no)
children = df['religiousness'].map({'yes':1,'no':0})

# actual rating (marriage happiness) likely in 'affairs' column (1-5)
rating = df['affairs']

for col in candidates:
    x = df[col]
    print("\nCandidate:", col)
    print("mean, std", x.mean(), x.std())
    # correlation with rating
    print("corr with rating", np.corrcoef(x, rating)[0,1])
    # t-test by children
    x_yes = x[children==1]
    x_no = x[children==0]
    t,p = stats.ttest_ind(x_yes, x_no, equal_var=False)
    print("mean yes/no", x_yes.mean(), x_no.mean(), "t", t, "p", p)

