import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.discrete.discrete_model import NegativeBinomial as NB2

csv_path = 'hurricane.csv'

df = pd.read_csv(csv_path)

print(df.head())
print(df.describe(include='all'))

# Coerce numeric columns used in models (guard against unexpected non-numeric values)
def num(col):
    return pd.to_numeric(df[col], errors='coerce')

# Outcome: deaths column appears to be 'name' (integer counts)
deaths = num('name')

# Femininity rating: column 'category' (1-11 scale, continuous)
masfem = num('category')

# Binary female: masfem_mturk
female_binary = num('masfem_mturk')

# Controls: intensity / strength
# wind speed appears to be 'year' (75-190)
wind_speed = num('year')

# min pressure appears to be 'ndam15' (909-1002)
min_pressure = num('ndam15')

# Saffir-Simpson category appears to be 'gender_mf' (1-5)
ss_cat = num('gender_mf')

# Year appears to be 'wind' (1950-2012)
year = num('wind')

# Damage (normalized) maybe 'elapsedyrs' or 'source'
# We'll include both separately in models to see robustness
dam1 = num('elapsedyrs')
dam2 = num('source')

# Basic correlations
mask_corr = masfem.notna() & deaths.notna()
print('\nPearson corr (masfem, deaths):', stats.pearsonr(masfem[mask_corr], deaths[mask_corr]))
print('Spearman corr (masfem, deaths):', stats.spearmanr(masfem[mask_corr], deaths[mask_corr]))

# log1p deaths
log_deaths = np.log1p(deaths)

# Simple OLS
simple_df = pd.DataFrame({'log_deaths': log_deaths, 'masfem': masfem}).dropna()
X_simple = sm.add_constant(simple_df['masfem'])
model_simple = sm.OLS(simple_df['log_deaths'], X_simple).fit()
print('\nSimple OLS log1p(deaths) ~ masfem')
print(model_simple.summary())

# Add controls (wind_speed, min_pressure, ss_cat, year)
controls_df = pd.DataFrame({
    'log_deaths': log_deaths,
    'masfem': masfem,
    'wind_speed': wind_speed,
    'min_pressure': min_pressure,
    'ss_cat': ss_cat,
    'year': year,
}).dropna()
X_controls = sm.add_constant(controls_df[['masfem', 'wind_speed', 'min_pressure', 'ss_cat', 'year']])
model_controls = sm.OLS(controls_df['log_deaths'], X_controls).fit()
print('\nOLS with controls log1p(deaths) ~ masfem + wind_speed + min_pressure + ss_cat + year')
print(model_controls.summary())

# Also try including damage variable dam1 (log1p) as additional control
controls_dam1_df = pd.DataFrame({
    'log_deaths': log_deaths,
    'masfem': masfem,
    'wind_speed': wind_speed,
    'min_pressure': min_pressure,
    'ss_cat': ss_cat,
    'year': year,
    'log_damage1': np.log1p(dam1),
}).replace([np.inf, -np.inf], np.nan).dropna()
X_controls_dam1 = sm.add_constant(controls_dam1_df[['masfem', 'wind_speed', 'min_pressure', 'ss_cat', 'year', 'log_damage1']])
model_controls_dam1 = sm.OLS(controls_dam1_df['log_deaths'], X_controls_dam1).fit()
print('\nOLS with controls + log(damage1) log1p(deaths) ~ ...')
print(model_controls_dam1.summary())

# Also try damage2
controls_dam2_df = pd.DataFrame({
    'log_deaths': log_deaths,
    'masfem': masfem,
    'wind_speed': wind_speed,
    'min_pressure': min_pressure,
    'ss_cat': ss_cat,
    'year': year,
    'log_damage2': np.log1p(dam2),
}).replace([np.inf, -np.inf], np.nan).dropna()
X_controls_dam2 = sm.add_constant(controls_dam2_df[['masfem', 'wind_speed', 'min_pressure', 'ss_cat', 'year', 'log_damage2']])
model_controls_dam2 = sm.OLS(controls_dam2_df['log_deaths'], X_controls_dam2).fit()
print('\nOLS with controls + log(damage2) log1p(deaths) ~ ...')
print(model_controls_dam2.summary())

# Poisson regression (counts) for robustness
pois_df = controls_df.copy()
X_pois = sm.add_constant(pois_df[['masfem', 'wind_speed', 'min_pressure', 'ss_cat', 'year']])
poisson = sm.GLM(pois_df['log_deaths'].apply(np.expm1), X_pois, family=sm.families.Poisson()).fit()
print('\nPoisson deaths ~ masfem + controls')
print(poisson.summary())

# Negative Binomial regression to address over-dispersion
nb = sm.GLM(pois_df['log_deaths'].apply(np.expm1), X_pois, family=sm.families.NegativeBinomial()).fit()
print('\nNegative Binomial deaths ~ masfem + controls')
print(nb.summary())

# Negative Binomial (NB2) with estimated dispersion
nb2 = NB2(pois_df['log_deaths'].apply(np.expm1), X_pois).fit(disp=0)
print('\nNegative Binomial (NB2) deaths ~ masfem + controls (alpha estimated)')
print(nb2.summary())

# Overdispersion check (Pearson chi2 / df)
overdispersion = poisson.pearson_chi2 / poisson.df_resid
print('\nPoisson overdispersion (Pearson chi2 / df):', overdispersion)

# Quick check of female_binary vs deaths
bin_df = pd.DataFrame({'log_deaths': log_deaths, 'female_binary': female_binary}).dropna()
X_bin = sm.add_constant(bin_df['female_binary'])
model_bin = sm.OLS(bin_df['log_deaths'], X_bin).fit()
print('\nSimple OLS log1p(deaths) ~ female_binary')
print(model_bin.summary())

# Save key results for review
print('\nKey coefficients:')
print('masfem simple OLS coef, p:', model_simple.params['masfem'], model_simple.pvalues['masfem'])
print('masfem controls OLS coef, p:', model_controls.params['masfem'], model_controls.pvalues['masfem'])
print('masfem controls+dam1 OLS coef, p:', model_controls_dam1.params['masfem'], model_controls_dam1.pvalues['masfem'])
print('masfem controls+dam2 OLS coef, p:', model_controls_dam2.params['masfem'], model_controls_dam2.pvalues['masfem'])
print('masfem Poisson coef, p:', poisson.params['masfem'], poisson.pvalues['masfem'])
print('masfem NegBin coef, p:', nb.params['masfem'], nb.pvalues['masfem'])
print('masfem NegBin2 coef, p:', nb2.params['masfem'], nb2.pvalues['masfem'])
