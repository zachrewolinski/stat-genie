from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/negative_leading_statement_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane-level dataframe into the final dataset used for modeling.

    Produced columns used in the statistical models (exact names):
      - masfem_scaled: standardized masfem score (z-score)
      - gender_mf: binary female-name indicator (0/1) (kept from raw data)
      - log_alldeaths: log1p(alldeaths)
      - log_ndam15: log1p(ndam15) (secondary outcome used in robustness checks)
      - severity_index: composite severity index (average of z-scored wind, inverse min pressure, and category)
      - year: original year (kept)
      - elapsedyrs: original elapsedyrs (kept)
      - source: original source (kept as categorical)

    Notes: rows with missing values in the core variables are dropped. Z-scores use sample mean & std.
    """
    df = df.copy()

    # Required columns for core analysis
    required_cols = ['alldeaths', 'ndam15', 'masfem', 'gender_mf', 'wind', 'min', 'category', 'year', 'elapsedyrs', 'source']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Drop rows with missing values in core variables
    df = df.dropna(subset=['alldeaths', 'masfem', 'wind', 'min', 'category', 'ndam15', 'gender_mf', 'year', 'elapsedyrs', 'source']).reset_index(drop=True)

    # Primary dependent variable: log(1 + fatalities)
    df['log_alldeaths'] = np.log1p(df['alldeaths'].astype(float))

    # Secondary outcome: log damages (normalized to 2015)
    df['log_ndam15'] = np.log1p(df['ndam15'].astype(float))

    # Standardize masfem and masfem_mturk (if present), and keep gender_mf as-is
    df['masfem_scaled'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0)
    if 'masfem_mturk' in df.columns:
        df['masfem_mturk_scaled'] = (df['masfem_mturk'] - df['masfem_mturk'].mean()) / (df['masfem_mturk'].std(ddof=0) if df['masfem_mturk'].std(ddof=0) != 0 else 1.0)

    # Keep binary gender variable (0/1) as provided
    df['gender_mf'] = df['gender_mf'].astype(int)

    # Construct severity index from wind, min pressure, and category.
    # Lower min pressure => more severe, so invert the z-score of min
    # Use population ddof=0 for standardization consistency in small sample
    wind_mean, wind_std = df['wind'].mean(), df['wind'].std(ddof=0) if df['wind'].std(ddof=0) != 0 else 1.0
    min_mean, min_std = df['min'].mean(), df['min'].std(ddof=0) if df['min'].std(ddof=0) != 0 else 1.0
    cat_mean, cat_std = df['category'].mean(), df['category'].std(ddof=0) if df['category'].std(ddof=0) != 0 else 1.0

    df['wind_z'] = (df['wind'] - wind_mean) / wind_std
    # invert min so that higher z means more severe (lower pressure -> more severe)
    df['min_z_inv'] = -((df['min'] - min_mean) / min_std)
    df['category_z'] = (df['category'] - cat_mean) / cat_std

    # Average the three z-scores to form a simple severity index
    df['severity_index'] = df[['wind_z', 'min_z_inv', 'category_z']].mean(axis=1)

    # Ensure source is categorical (kept as original string categories)
    df['source'] = df['source'].astype('category')

    # Keep only columns required for modeling plus a few useful diagnostics
    keep_cols = [
        'alldeaths', 'log_alldeaths', 'ndam15', 'log_ndam15',
        'masfem', 'masfem_scaled', 'gender_mf',
        'severity_index', 'wind', 'min', 'category',
        'year', 'elapsedyrs', 'source'
    ]
    # masfem_mturk_scaled optional
    if 'masfem_mturk_scaled' in df.columns:
        keep_cols.append('masfem_mturk_scaled')

    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit multiple model specifications to test whether more-feminine hurricane names (higher masfem)
    predict fewer precautionary outcomes (proxied here by fatalities and property damage).

    Models implemented:
      1) OLS on log_alldeaths with robust (HC3) standard errors (primary specification)
      2) Negative binomial GLM on raw alldeaths (count outcome) with log link (robust to overdispersion)
      3) OLS on log_ndam15 (robustness using damage as alternative outcome)
      4) Alternate specifications replacing masfem_scaled with gender_mf (binary)

    Returns a dictionary of fitted model results objects and textual summaries.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    results = {}

    # Basic checks
    required = ['log_alldeaths', 'alldeaths', 'log_ndam15', 'masfem_scaled', 'gender_mf', 'severity_index', 'year', 'elapsedyrs', 'source']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Transformed dataframe is missing required columns for modeling: {missing}")

    # Primary OLS specification (log fatalities)
    formula_ols = 'log_alldeaths ~ masfem_scaled + severity_index + year + elapsedyrs + C(source)'
    ols_masfem = smf.ols(formula_ols, data=df).fit(cov_type='HC3')
    results['ols_masfem'] = {
        'model': ols_masfem,
        'summary_text': ols_masfem.summary().as_text()
    }

    # Negative binomial GLM on raw counts of fatalities (alldeaths)
    # Use same regressors; GLM will estimate coefficients on log link by default for NB in statsmodels
    formula_nb = 'alldeaths ~ masfem_scaled + severity_index + year + elapsedyrs + C(source)'
    try:
        nb_masfem = smf.glm(formula_nb, data=df, family=sm.families.NegativeBinomial()).fit()
        results['nb_masfem'] = {
            'model': nb_masfem,
            'summary_text': nb_masfem.summary().as_text()
        }
    except Exception as e:
        # If the NegativeBinomial family call fails (rare), capture the error
        results['nb_masfem_error'] = str(e)

    # Robustness: OLS on log damages (ndam15)
    formula_ols_damage = 'log_ndam15 ~ masfem_scaled + severity_index + year + elapsedyrs + C(source)'
    ols_damage = smf.ols(formula_ols_damage, data=df).fit(cov_type='HC3')
    results['ols_masfem_damage'] = {
        'model': ols_damage,
        'summary_text': ols_damage.summary().as_text()
    }

    # Alternate IV: binary gender indicator
    formula_ols_bin = 'log_alldeaths ~ gender_mf + severity_index + year + elapsedyrs + C(source)'
    ols_gender = smf.ols(formula_ols_bin, data=df).fit(cov_type='HC3')
    results['ols_gender'] = {
        'model': ols_gender,
        'summary_text': ols_gender.summary().as_text()
    }

    # Alternate NB with binary indicator
    formula_nb_bin = 'alldeaths ~ gender_mf + severity_index + year + elapsedyrs + C(source)'
    try:
        nb_gender = smf.glm(formula_nb_bin, data=df, family=sm.families.NegativeBinomial()).fit()
        results['nb_gender'] = {
            'model': nb_gender,
            'summary_text': nb_gender.summary().as_text()
        }
    except Exception as e:
        results['nb_gender_error'] = str(e)

    # Return fitted models and textual summaries so downstream code (or an analyst) can inspect significance and coefficients.
    return results


