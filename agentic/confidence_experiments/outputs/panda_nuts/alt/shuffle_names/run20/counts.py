import pandas as pd

df = pd.read_csv('panda_nuts.csv')
analysis = df.copy()
analysis['age_years'] = analysis['hammer']
analysis['sex'] = analysis['nuts_opened']
analysis['helped'] = analysis['seconds'].str.upper().map({'Y': 1, 'YES': 1, 'N': 0})
analysis['nuts_opened_count'] = analysis['help']
analysis['session_seconds'] = analysis['chimpanzee']
analysis['efficiency'] = analysis['nuts_opened_count'] / analysis['session_seconds']

print('sex counts')
print(analysis['sex'].value_counts())
print('help counts')
print(analysis['helped'].value_counts())
