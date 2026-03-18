import pandas as pd
import numpy as np

df = pd.read_csv('affairs.csv')

# map candidate predictor columns
years_married = df['children']  # values 0.125-15
age_bins = df['occupation']     # 17.5-57
religious_1_5 = df['rating']    # 1-5 (one of them)
marriage_rating = df['affairs'] # 1-5 (other)

for outcome in ['age', 'education', 'affairs', 'rating']:
    corr_yrs = np.corrcoef(df[outcome], years_married)[0,1]
    corr_age = np.corrcoef(df[outcome], age_bins)[0,1]
    corr_rel = np.corrcoef(df[outcome], religious_1_5)[0,1]
    corr_mar = np.corrcoef(df[outcome], marriage_rating)[0,1]
    print(outcome, 'corr yearsmarried', corr_yrs, 'corr age', corr_age, 'corr rel', corr_rel, 'corr marriage', corr_mar)
