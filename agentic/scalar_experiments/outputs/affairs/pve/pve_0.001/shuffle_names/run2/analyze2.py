import pandas as pd
import numpy as np

df = pd.read_csv('affairs.csv')

educ = df['education']/1000
print('Rounded education/1000 counts:')
print(educ.round().value_counts().sort_index())

age = df['age']
print('\nRounded age counts:')
print(age.round().value_counts().sort_index().head(20))

# Check if education/1000 within 0-12 mostly
print('\nEducation/1000 range', educ.min(), educ.max())
print('Education/1000 <=12 proportion', (educ<=12).mean())

# Check if age within 0-12 proportion
print('Age <=12 proportion', (age<=12).mean(), 'age>=0 proportion', (age>=0).mean())
