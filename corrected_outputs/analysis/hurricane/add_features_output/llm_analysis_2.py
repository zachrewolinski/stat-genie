from typing import Any, Dict
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/hurricane/add_features_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare and clean the hurricane dataset for modeling.

    This function:
    - Makes sure numeric columns are numeric (coerce errors to NaN)
    - Creates a binary female-name indicator from gender_mf
    - Creates log-transformed outcome variables (log_deaths, log_ndam15)
    - Standardizes masfem and masfem_mturk into z-scores (masfem_z, masfem_mturk_z)
    - Drops rows with missing values in the core model columns
    - Ensures required final columns are numeric (float64)
    - Returns the dataframe with added columns (and filtered rows)
    """
    df = df.copy()

    # Ensure numeric types for columns used in modeling (coerce non-numeric)
    numeric_cols = [
        'masfem', 'masfem_mturk', 'gender_mf', 'alldeaths', 'ndam15',
        'wind', 'min', 'category', 'year', 'elapsedyrs'
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Binary indicator for female name (use provided gender_mf; 1 female, 0 male)
    # Keep as numeric float (0.0/1.0) so statsmodels will accept dtype
    if 'gender_mf' in df.columns:
        df['female_name'] = pd.to_numeric(df['gender_mf'], errors='coerce').astype('float64')
    else:
        df['female_name'] = np.nan

    # Outcomes: log-transform fatalities and 2015-adjusted damage as robustness
    if 'alldeaths' in df.columns:
        # ensure alldeaths numeric, fill NaN with 0 for log transformation as intended
        df['alldeaths'] = pd.to_numeric(df['alldeaths'], errors='coerce').fillna(0).astype('float64')
        df['log_deaths'] = np.log(df['alldeaths'] + 1)
    else:
        df['log_deaths'] = np.nan

    if 'ndam15' in df.columns:
        df['ndam15'] = pd.to_numeric(df['ndam15'], errors='coerce')
        df['log_ndam15'] = np.log(df['ndam15'].fillna(0) + 1)
    else:
        df['log_ndam15'] = np.nan

    # Standardize (z-score) masfem and masfem_mturk for interpretability
    if 'masfem' in df.columns:
        mean_m = df['masfem'].mean()
        std_m = df['masfem'].std()
        if pd.isna(std_m) or std_m == 0:
            df['masfem_z'] = np.nan
        else:
            df['masfem_z'] = (df['masfem'] - mean_m) / std_m
    else:
        df['masfem_z'] = np.nan

    if 'masfem_mturk' in df.columns:
        mean_mt = df['masfem_mturk'].mean()
        std_mt = df['masfem_mturk'].std()
        if pd.isna(std_mt) or std_mt == 0:
            df['masfem_mturk_z'] = np.nan
        else:
            df['masfem_mturk_z'] = (df['masfem_mturk'] - mean_mt) / std_mt
    else:
        df['masfem_mturk_z'] = np.nan

    # Required final columns for modeling (must exist in final dataframe)
    required_cols = [
        'masfem_z', 'female_name', 'log_deaths',
        'wind', 'category', 'min', 'year', 'elapsedyrs'
    ]

    # Keep only required columns that actually exist in the dataframe for NA-dropping.
    required_cols_existing = [c for c in required_cols if c in df.columns]

    # Drop rows missing any of the required model columns
    if required_cols_existing:
        df = df.dropna(subset=required_cols_existing)

    # Ensure required columns have numeric dtype (float64) for statsmodels compatibility
    for c in required_cols_existing:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')

    # Also ensure auxiliary columns used in models are numeric if present
    if 'alldeaths' in df.columns:
        df['alldeaths'] = pd.to_numeric(df['alldeaths'], errors='coerce').astype('float64')
    if 'log_ndam15' in df.columns:
        df['log_ndam15'] = pd.to_numeric(df['log_ndam15'], errors='coerce').astype('float64')

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Fit statistical models to test whether more feminine hurricane names are associated
    with fewer fatalities (proxy for reduced precautionary behavior), controlling for
    storm severity and time trends.

    Models fitted:
    - OLS on log_deaths (log(alldeaths + 1)) with robust (HC1) SEs
    - Negative binomial GLM on raw alldeaths counts (robust to overdispersion)
    - OLS on log_ndam15 (log damage) as a robustness check

    Returns a dictionary with fitted model results objects.
    """
    results: Dict[str, Any] = {}

    # Covariates for the main specification (must match conceptual variable column names)
    X_cols = ['masfem_z', 'female_name', 'wind', 'category', 'min', 'year', 'elapsedyrs']
    # Keep only columns present in the dataframe
    X_cols = [c for c in X_cols if c in df.columns]

    if len(X_cols) == 0:
        raise ValueError('No covariates available in dataframe for modeling.')

    # Prepare design matrix and ensure numeric types
    X = df[X_cols].copy()
    X = X.apply(pd.to_numeric, errors='coerce').astype(float)

    # Add constant term
    X = sm.add_constant(X, has_constant='add')

    # Helper to align X and y (drop rows with NA in either)
    def align_and_clean(y_series: pd.Series, X_df: pd.DataFrame):
        y = y_series.copy()
        y = pd.to_numeric(y, errors='coerce').astype(float)
        model_df = pd.concat([y.rename('_y'), X_df], axis=1).dropna()
        y_clean = model_df['_y']
        X_clean = model_df.drop(columns=['_y'])
        return y_clean, X_clean

    # 1) OLS on log_deaths
    if 'log_deaths' in df.columns:
        y = df['log_deaths']
        y_clean, X_clean = align_and_clean(y, X)
        if X_clean.shape[0] == 0:
            results['ols_log_deaths'] = None
        else:
            ols_model = sm.OLS(y_clean, X_clean).fit(cov_type='HC1')
            results['ols_log_deaths'] = ols_model
    else:
        results['ols_log_deaths'] = None

    # 2) Negative binomial on count of deaths (alldeaths)
    if 'alldeaths' in df.columns:
        y_nb = df['alldeaths']
        y_nb_clean, X_nb_clean = align_and_clean(y_nb, X)
        if X_nb_clean.shape[0] == 0:
            results['nb_deaths'] = None
        else:
            try:
                nb_model = sm.GLM(y_nb_clean, X_nb_clean, family=sm.families.NegativeBinomial()).fit()
                results['nb_deaths'] = nb_model
            except Exception:
                # fallback to Poisson if NB fails for any reason
                poisson_model = sm.GLM(y_nb_clean, X_nb_clean, family=sm.families.Poisson()).fit()
                results['nb_deaths'] = poisson_model
    else:
        results['nb_deaths'] = None

    # 3) Robustness: OLS on log adjusted damage (ndam15)
    if 'log_ndam15' in df.columns:
        y2 = df['log_ndam15']
        y2_clean, X2_clean = align_and_clean(y2, X)
        if X2_clean.shape[0] == 0:
            results['ols_log_damage'] = None
        else:
            ols_model_damage = sm.OLS(y2_clean, X2_clean).fit(cov_type='HC1')
            results['ols_log_damage'] = ols_model_damage
    else:
        results['ols_log_damage'] = None

    return results