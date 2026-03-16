import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = 'hurricane.csv'

df = pd.read_csv(DATA_PATH)

# Map features to meaningful names
col_map = {
    'feature1': 'id',
    'feature2': 'year',
    'feature3': 'name',
    'feature4': 'masfem',
    'feature5': 'min_pressure',
    'feature6': 'female_binary',
    'feature7': 'category',
    'feature8': 'deaths',
    'feature9': 'damage_2013',
    'feature10': 'years_since',
    'feature11': 'source',
    'feature12': 'masfem_mt',
    'feature13': 'max_wind',
    'feature14': 'damage_2015',
}

df = df.rename(columns=col_map)

# Basic cleaning: ensure numeric types
for col in ['masfem', 'masfem_mt', 'min_pressure', 'category', 'deaths', 'damage_2013', 'damage_2015', 'max_wind']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Add transformed outcomes to handle skew
# Use log1p for deaths and damages
for col in ['deaths', 'damage_2013', 'damage_2015']:
    df[f'log1p_{col}'] = np.log1p(df[col])

# Drop rows with missing key variables
key_cols = ['masfem', 'deaths', 'min_pressure', 'max_wind', 'category']
analysis_df = df.dropna(subset=key_cols).copy()

summary = {
    'n_total': int(len(df)),
    'n_analysis': int(len(analysis_df)),
    'missing_masfem': int(df['masfem'].isna().sum()),
    'missing_deaths': int(df['deaths'].isna().sum()),
}

# Simple correlations
corr_deaths = analysis_df[['masfem', 'deaths']].corr().iloc[0,1]
corr_log_deaths = analysis_df[['masfem', 'log1p_deaths']].corr().iloc[0,1]

# OLS: log deaths ~ masfem + intensity controls
# Controls: min_pressure (lower is stronger), max_wind, category
ols_model = smf.ols('log1p_deaths ~ masfem + min_pressure + max_wind + category', data=analysis_df).fit()

# Alternative: use female_binary instead of masfem
ols_binary = smf.ols('log1p_deaths ~ female_binary + min_pressure + max_wind + category', data=analysis_df).fit()

# Interaction models: effect of name femininity conditional on severity
ols_interaction_pressure = smf.ols(
    'log1p_deaths ~ masfem * min_pressure + max_wind + category',
    data=analysis_df
).fit()
ols_interaction_wind = smf.ols(
    'log1p_deaths ~ masfem * max_wind + min_pressure + category',
    data=analysis_df
).fit()

# Also test damage as an outcome (proxy for risk exposure; not a direct measure of precautions)
ols_damage = smf.ols('log1p_damage_2015 ~ masfem + min_pressure + max_wind + category', data=analysis_df).fit()

result = {
    'summary': summary,
    'corr_deaths': float(corr_deaths),
    'corr_log_deaths': float(corr_log_deaths),
    'ols_log_deaths': {
        'params': ols_model.params.to_dict(),
        'pvalues': ols_model.pvalues.to_dict(),
        'r2': float(ols_model.rsquared),
        'nobs': int(ols_model.nobs),
    },
    'ols_log_deaths_interaction_pressure': {
        'params': ols_interaction_pressure.params.to_dict(),
        'pvalues': ols_interaction_pressure.pvalues.to_dict(),
        'r2': float(ols_interaction_pressure.rsquared),
        'nobs': int(ols_interaction_pressure.nobs),
    },
    'ols_log_deaths_interaction_wind': {
        'params': ols_interaction_wind.params.to_dict(),
        'pvalues': ols_interaction_wind.pvalues.to_dict(),
        'r2': float(ols_interaction_wind.rsquared),
        'nobs': int(ols_interaction_wind.nobs),
    },
    'ols_log_deaths_binary': {
        'params': ols_binary.params.to_dict(),
        'pvalues': ols_binary.pvalues.to_dict(),
        'r2': float(ols_binary.rsquared),
        'nobs': int(ols_binary.nobs),
    },
    'ols_log_damage_2015': {
        'params': ols_damage.params.to_dict(),
        'pvalues': ols_damage.pvalues.to_dict(),
        'r2': float(ols_damage.rsquared),
        'nobs': int(ols_damage.nobs),
    }
}

print(json.dumps(result, indent=2, sort_keys=True))
