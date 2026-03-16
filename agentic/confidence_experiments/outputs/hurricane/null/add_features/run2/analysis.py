import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('hurricane.csv')

# Ensure numeric types
numeric_cols = [
    'masfem','masfem_mturk','gender_mf','alldeaths','wind','min','category','ndam','ndam15','year','elapsedyrs'
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing key variables
key_cols = ['masfem','alldeaths','wind','min','category']
df_key = df.dropna(subset=key_cols).copy()

# Transform deaths
# Use log1p to handle zeros

df_key['log_deaths'] = np.log1p(df_key['alldeaths'])

# Simple correlation
corr = df_key[['masfem','alldeaths']].corr().iloc[0,1]

# Male vs female mean deaths
mean_deaths = df_key.groupby('gender_mf')['alldeaths'].mean()
count_by_gender = df_key['gender_mf'].value_counts()

# OLS regression log_deaths on masfem + controls
X_cols = ['masfem','wind','min','category','year']
X = sm.add_constant(df_key[X_cols])
ols_model = sm.OLS(df_key['log_deaths'], X, missing='drop').fit()

# OLS with gender_mf instead of masfem
X_cols2 = ['gender_mf','wind','min','category','year']
X2 = sm.add_constant(df_key[X_cols2])
ols_model_gender = sm.OLS(df_key['log_deaths'], X2, missing='drop').fit()

# Interaction models (masfem * wind)
ols_interaction = smf.ols(
    'log_deaths ~ masfem + wind + min + category + year + masfem:wind',
    data=df_key
).fit()

ols_interaction_gender = smf.ols(
    'log_deaths ~ gender_mf + wind + min + category + year + gender_mf:wind',
    data=df_key
).fit()

# Negative binomial GLM on deaths
nb_model = sm.GLM(
    df_key['alldeaths'],
    X,
    family=sm.families.NegativeBinomial()
).fit()

nb_model_gender = sm.GLM(
    df_key['alldeaths'],
    X2,
    family=sm.families.NegativeBinomial()
).fit()

# Interaction NB models (masfem * wind)
nb_interaction = smf.glm(
    'alldeaths ~ masfem + wind + min + category + year + masfem:wind',
    data=df_key,
    family=sm.families.NegativeBinomial()
).fit()

nb_interaction_gender = smf.glm(
    'alldeaths ~ gender_mf + wind + min + category + year + gender_mf:wind',
    data=df_key,
    family=sm.families.NegativeBinomial()
).fit()

results = {
    'n': int(len(df_key)),
    'corr_masfem_deaths': corr,
    'mean_deaths_male': float(mean_deaths.get(0, np.nan)),
    'mean_deaths_female': float(mean_deaths.get(1, np.nan)),
    'count_male': int(count_by_gender.get(0, 0)),
    'count_female': int(count_by_gender.get(1, 0)),
    'ols_masfem_coef': float(ols_model.params.get('masfem', np.nan)),
    'ols_masfem_p': float(ols_model.pvalues.get('masfem', np.nan)),
    'ols_gender_coef': float(ols_model_gender.params.get('gender_mf', np.nan)),
    'ols_gender_p': float(ols_model_gender.pvalues.get('gender_mf', np.nan)),
    'ols_interaction_masfem_wind_coef': float(ols_interaction.params.get('masfem:wind', np.nan)),
    'ols_interaction_masfem_wind_p': float(ols_interaction.pvalues.get('masfem:wind', np.nan)),
    'ols_interaction_gender_wind_coef': float(ols_interaction_gender.params.get('gender_mf:wind', np.nan)),
    'ols_interaction_gender_wind_p': float(ols_interaction_gender.pvalues.get('gender_mf:wind', np.nan)),
    'nb_masfem_coef': float(nb_model.params.get('masfem', np.nan)),
    'nb_masfem_p': float(nb_model.pvalues.get('masfem', np.nan)),
    'nb_gender_coef': float(nb_model_gender.params.get('gender_mf', np.nan)),
    'nb_gender_p': float(nb_model_gender.pvalues.get('gender_mf', np.nan)),
    'nb_interaction_masfem_wind_coef': float(nb_interaction.params.get('masfem:wind', np.nan)),
    'nb_interaction_masfem_wind_p': float(nb_interaction.pvalues.get('masfem:wind', np.nan)),
    'nb_interaction_gender_wind_coef': float(nb_interaction_gender.params.get('gender_mf:wind', np.nan)),
    'nb_interaction_gender_wind_p': float(nb_interaction_gender.pvalues.get('gender_mf:wind', np.nan)),
}

with open('analysis_results.json','w') as f:
    json.dump(results,f,indent=2)

print(json.dumps(results, indent=2))
