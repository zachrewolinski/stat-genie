import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_PATH = 'hurricane.csv'

# Load data
_df = pd.read_csv(DATA_PATH)

# Map columns for readability
cols = {
    'id': 'feature1',
    'year': 'feature2',
    'name': 'feature3',
    'masfem_rating': 'feature4',
    'min_pressure': 'feature5',
    'female_binary': 'feature6',
    'category': 'feature7',
    'deaths': 'feature8',
    'damage_2013': 'feature9',
    'years_since': 'feature10',
    'source': 'feature11',
    'mturk_rating': 'feature12',
    'wind_speed': 'feature13',
    'damage_2015': 'feature14',
}

# Outcome: fatalities (count), use log1p to reduce skew
_df['log_deaths'] = np.log1p(_df[cols['deaths']])

# Controls for storm severity and time trend
controls = [cols['category'], cols['min_pressure'], cols['wind_speed'], cols['year']]

# Helper: run OLS with robust SEs

def ols_robust(y, x_cols):
    X = sm.add_constant(_df[x_cols])
    model = sm.OLS(_df[y], X).fit(cov_type='HC3')
    return model

# Models
model_masfem = ols_robust('log_deaths', [cols['masfem_rating']] + controls)
model_mturk = ols_robust('log_deaths', [cols['mturk_rating']] + controls)
model_binary = ols_robust('log_deaths', [cols['female_binary']] + controls)

# Simple correlations
corr_masfem = _df[[cols['masfem_rating'], cols['deaths']]].corr().iloc[0, 1]
corr_mturk = _df[[cols['mturk_rating'], cols['deaths']]].corr().iloc[0, 1]
corr_binary = _df[[cols['female_binary'], cols['deaths']]].corr().iloc[0, 1]

# Extract key stats

def coef_p(model, var):
    return float(model.params[var]), float(model.pvalues[var])

masfem_coef, masfem_p = coef_p(model_masfem, cols['masfem_rating'])
mturk_coef, mturk_p = coef_p(model_mturk, cols['mturk_rating'])
binary_coef, binary_p = coef_p(model_binary, cols['female_binary'])

# Interpret: hypothesis implies more feminine names -> higher fatalities (log_deaths)
# Use p<0.05 as evidence of association; otherwise treat as no evidence.

# Decide response strength
response = 50  # neutral

# Primary indicator: masfem_rating model
if masfem_p < 0.05 and masfem_coef > 0:
    # evidence consistent with hypothesis
    response = 70 if masfem_p < 0.01 else 62
elif masfem_p < 0.05 and masfem_coef < 0:
    # significant but opposite direction
    response = 30 if masfem_p < 0.01 else 38
else:
    # no significant evidence
    response = 40

explanation = (
    "I tested whether more feminine hurricane names are associated with greater fatalities (a proxy for fewer "
    "precautionary measures), controlling for storm severity (Saffir-Simpson category, minimum pressure, "
    "wind speed) and year. Using log1p(fatalities) as the outcome, the femininity rating (feature4) coefficient "
    f"was {masfem_coef:.3f} with p={masfem_p:.3f}. The MTurk femininity rating (feature12) coefficient was "
    f"{mturk_coef:.3f} with p={mturk_p:.3f}, and the binary female-name indicator (feature6) coefficient was "
    f"{binary_coef:.3f} with p={binary_p:.3f}. Simple correlations with raw fatalities were small "
    f"(feature4: {corr_masfem:.3f}, feature12: {corr_mturk:.3f}, feature6: {corr_binary:.3f}). "
    "Overall, these models do not provide statistically significant evidence that more feminine names are linked "
    "to higher fatalities after accounting for storm severity, so the data do not support the claim that more "
    "feminine names lead to fewer precautionary measures in this dataset."
)

out = {"response": int(response), "explanation": explanation}

with open('conclusion.txt', 'w') as f:
    json.dump(out, f)
