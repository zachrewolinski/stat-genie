import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Rename columns to inferred meanings
# based on distributions and sample values
rename = {
    'ndam': 'id',
    'wind': 'year',
    'alldeaths': 'storm_name',
    'category': 'feminity_rating',  # 1-11 scale
    'ndam15': 'min_pressure',
    'masfem_mturk': 'female_binary',
    'gender_mf': 'ss_category',
    'name': 'deaths',
    'elapsedyrs': 'damage_2013',
    'masfem': 'years_elapsed',
    'min': 'data_source',
    'ind': 'mturk_rating',
    'year': 'max_wind',
    'source': 'damage_2015',
}

df = df.rename(columns=rename)

# Basic transformations
for col in ['deaths', 'damage_2013', 'damage_2015']:
    df[f'log_{col}'] = np.log1p(df[col])

# Prepare a regression function

def ols_summary(y, X, label):
    X = sm.add_constant(X)
    model = sm.OLS(y, X, missing='drop').fit()
    print(f"\n{label}")
    print(model.summary().tables[1])
    return model

# Core controls for storm intensity
controls = ['ss_category', 'max_wind', 'min_pressure', 'year']

# Regress log deaths on femininity rating
model1 = ols_summary(df['log_deaths'], df[['feminity_rating'] + controls],
                    'log_deaths ~ femininity_rating + intensity controls')

# Regress log deaths on binary female indicator
model2 = ols_summary(df['log_deaths'], df[['female_binary'] + controls],
                    'log_deaths ~ female_binary + intensity controls')

# Regress log damage on femininity rating
model3 = ols_summary(df['log_damage_2013'], df[['feminity_rating'] + controls],
                    'log_damage_2013 ~ femininity_rating + intensity controls')

# Regress log damage on female_binary
model4 = ols_summary(df['log_damage_2013'], df[['female_binary'] + controls],
                    'log_damage_2013 ~ female_binary + intensity controls')

# Simple correlations
print('\nCorrelations (pairwise):')
print('feminity_rating vs log_deaths:', df[['feminity_rating','log_deaths']].corr().iloc[0,1])
print('mturk_rating vs log_deaths:', df[['mturk_rating','log_deaths']].corr().iloc[0,1])
print('female_binary vs log_deaths:', df[['female_binary','log_deaths']].corr().iloc[0,1])

print('feminity_rating vs log_damage_2013:', df[['feminity_rating','log_damage_2013']].corr().iloc[0,1])
print('female_binary vs log_damage_2013:', df[['female_binary','log_damage_2013']].corr().iloc[0,1])

# Save key outputs for reference
results = {
    'model1_feminity_coef': model1.params.get('feminity_rating'),
    'model1_feminity_p': model1.pvalues.get('feminity_rating'),
    'model2_female_coef': model2.params.get('female_binary'),
    'model2_female_p': model2.pvalues.get('female_binary'),
    'model3_feminity_coef': model3.params.get('feminity_rating'),
    'model3_feminity_p': model3.pvalues.get('feminity_rating'),
    'model4_female_coef': model4.params.get('female_binary'),
    'model4_female_p': model4.pvalues.get('female_binary'),
}

print('\nKey results:', results)
