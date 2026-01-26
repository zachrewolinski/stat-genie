from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/negative_leading_statement_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the hurricane dataset for modeling.

    Produces the following new/renamed columns used in the models (exact names used in modeling):
      - NameFem_masf: continuous coder-based femininity rating (from column 'masfem')
      - NameFem_mturk: continuous MTurk-based femininity rating (from 'masfem_mturk')
      - NameFem_bin: binary gender indicator of the name (from 'gender_mf')
      - Deaths: integer count of fatalities (from 'alldeaths')
      - log_ndam15: log(ndam15 + 1) to reduce skew in damage (ndam15 is already inflation/wealth/pop adjusted)
      - wind, min, category: preserved from original for reference
      - Intensity_z: z-scored composite intensity index combining wind, inverted min pressure, and category
      - YearC: centered year (year - mean(year))
      - ElapsedYears: copy of 'elapsedyrs'
      - source: preserved categorical source column

    The function drops rows missing any of the critical variables used in models.
    """
    df = df.copy()

    # Required original columns
    required_cols = [
        'masfem', 'masfem_mturk', 'gender_mf', 'alldeaths', 'ndam15',
        'wind', 'min', 'category', 'elapsedyrs', 'year', 'source'
    ]

    # Drop rows missing any required variables (we keep only storms with complete info for the primary specs)
    df = df.dropna(subset=required_cols)

    # Create explicit IV columns (use the original ratings but with clear names)
    df['NameFem_masf'] = df['masfem'].astype(float)
    df['NameFem_mturk'] = df['masfem_mturk'].astype(float)
    # Binary gender indicator (0 male, 1 female) - ensure integer
    # If gender_mf is already numeric-coded, use it; otherwise attempt to map common string codes
    try:
        df['NameFem_bin'] = df['gender_mf'].astype(int)
    except Exception:
        # fallback mapping: common encodings
        gender_map = {'M': 0, 'Male': 0, 'F': 1, 'Female': 1}
        df['NameFem_bin'] = df['gender_mf'].map(gender_map)
        df['NameFem_bin'] = df['NameFem_bin'].astype(int)

    # Dependent variables
    # Deaths: keep as integer count
    df['Deaths'] = df['alldeaths'].astype(int)
    # Log transform damage to reduce skew; ndam15 is already adjusted in dataset
    df['log_ndam15'] = np.log(df['ndam15'].astype(float) + 1.0)

    # Controls: keep raw physical measures
    df['wind'] = df['wind'].astype(float)
    df['min'] = df['min'].astype(float)
    df['category'] = df['category'].astype(float)

    # Year centered
    df['YearC'] = df['year'].astype(float) - df['year'].astype(float).mean()

    # Elapsed years (as provided)
    df['ElapsedYears'] = df['elapsedyrs'].astype(float)

    # Create a composite intensity index (z-score average of wind, inverted min pressure, and category)
    # Higher Intensity_z => stronger/more dangerous storm
    # invert min pressure because lower pressure => stronger storm
    wind_std = df['wind'].std(ddof=0)
    wind_z = (df['wind'] - df['wind'].mean()) / (wind_std if wind_std != 0 else 1.0)
    negmin = -df['min']
    negmin_std = negmin.std(ddof=0)
    negmin_z = (negmin - negmin.mean()) / (negmin_std if negmin_std != 0 else 1.0)
    category_std = df['category'].std(ddof=0)
    category_z = (df['category'] - df['category'].mean()) / (category_std if category_std != 0 else 1.0)

    df['Intensity_z'] = (wind_z + negmin_z + category_z) / 3.0

    # Preserve source categorical variable as-is (user can convert to dummies in modeling step if desired)
    df['source'] = df['source'].astype('category')

    # Final check: drop any rows that produced non-finite values (just in case)
    df = df[np.isfinite(df['Intensity_z']) & np.isfinite(df['log_ndam15'])]

    # Reset index for convenience
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Runs the primary statistical analyses to test whether more feminine hurricane names are associated with
    (a) fewer deaths and (b) lower economic damage, after controlling for storm intensity and time.

    Models run (main specifications):
      1) Negative binomial GLM for Deaths with NameFem_masf as IV and Intensity_z, YearC, ElapsedYears as controls.
      2) OLS regression for log_ndam15 with the same covariates.

    Robustness/specification checks (also returned):
      - Replace NameFem_masf with NameFem_mturk (MTurk rating)
      - Replace NameFem_masf with NameFem_bin (binary female name indicator)

    Returns a dictionary of fitted results (statsmodels results objects). Users can inspect .summary() for each.
    """
    results = {}

    # Ensure the dataframe passed in contains the transformed columns
    required = ['Deaths', 'log_ndam15', 'NameFem_masf', 'NameFem_mturk', 'NameFem_bin', 'Intensity_z', 'YearC', 'ElapsedYears']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Dataframe is missing required columns for modeling: {missing}")

    # -- Primary negative binomial model for Deaths (count, overdispersed) --
    formula_nb = 'Deaths ~ NameFem_masf + Intensity_z + YearC + ElapsedYears'
    # Fit GLM with Negative Binomial family and request HC3 robust covariance via fit argument
    nb_fit = smf.glm(formula_nb, data=df, family=sm.families.NegativeBinomial()).fit(cov_type='HC3')
    # Store fitted result (fit called with cov_type='HC3' provides robust covariance)
    results['nb_main_masf'] = nb_fit

    # -- OLS for logged damage --
    formula_ols = 'log_ndam15 ~ NameFem_masf + Intensity_z + YearC + ElapsedYears'
    ols_fit = smf.ols(formula_ols, data=df).fit(cov_type='HC3')
    results['ols_damage_main_masf'] = ols_fit

    # -- Robustness 1: use MTurk-rated femininity --
    formula_nb_mturk = 'Deaths ~ NameFem_mturk + Intensity_z + YearC + ElapsedYears'
    nb_mturk = smf.glm(formula_nb_mturk, data=df, family=sm.families.NegativeBinomial()).fit(cov_type='HC3')
    results['nb_mturk'] = nb_mturk

    formula_ols_mturk = 'log_ndam15 ~ NameFem_mturk + Intensity_z + YearC + ElapsedYears'
    ols_mturk = smf.ols(formula_ols_mturk, data=df).fit(cov_type='HC3')
    results['ols_damage_mturk'] = ols_mturk

    # -- Robustness 2: binary female name indicator --
    formula_nb_bin = 'Deaths ~ NameFem_bin + Intensity_z + YearC + ElapsedYears'
    nb_bin = smf.glm(formula_nb_bin, data=df, family=sm.families.NegativeBinomial()).fit(cov_type='HC3')
    results['nb_bin'] = nb_bin

    formula_ols_bin = 'log_ndam15 ~ NameFem_bin + Intensity_z + YearC + ElapsedYears'
    ols_bin = smf.ols(formula_ols_bin, data=df).fit(cov_type='HC3')
    results['ols_damage_bin'] = ols_bin

    # Return the dictionary of model results. Each value is a statsmodels results object; call .summary() to view.
    return results