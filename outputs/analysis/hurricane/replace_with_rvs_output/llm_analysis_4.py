from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure numeric columns are numeric (coerce invalid entries to NaN)
    numeric_cols = ['alldeaths', 'ndam15', 'masfem', 'gender_mf', 'wind', 'min', 'category', 'year', 'elapsedyrs']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing the primary dependent variable or primary IV or key controls
    required = [c for c in ['alldeaths', 'masfem', 'wind', 'min', 'category', 'year'] if c in df.columns]
    if len(required) > 0:
        df = df.dropna(subset=required)

    # Dependent variables: log-transform to reduce skew and handle zeros
    if 'alldeaths' in df.columns:
        df['log_alldeaths'] = np.log1p(df['alldeaths'])

    # Property damage alternative DV (log-transformed 2015-normalized damage)
    if 'ndam15' in df.columns:
        df['log_ndam15'] = np.log1p(df['ndam15'])

    # Independent variable: standardized masfem (mean 0, SD 1)
    if 'masfem' in df.columns:
        mean_m = df['masfem'].mean(skipna=True)
        sd_m = df['masfem'].std(ddof=0, skipna=True)
        if pd.isna(mean_m):
            mean_m = 0.0
        if pd.isna(sd_m) or sd_m == 0:
            sd_m = 1.0
        df['masfem_scaled'] = (df['masfem'] - mean_m) / sd_m

    # Binary gender indicator from provided binary variable (0 male, 1 female)
    if 'gender_mf' in df.columns:
        # Convert to numeric (coerce invalid entries), keep as float so statsmodels accepts it cleanly
        df['gender_female'] = pd.to_numeric(df['gender_mf'], errors='coerce').astype('float64')

    # Center year to ease interpretation
    if 'year' in df.columns:
        year_mean = df['year'].mean(skipna=True)
        if pd.isna(year_mean):
            year_mean = 0.0
        df['year_c'] = df['year'] - year_mean

    # Ensure final model columns are numeric dtypes where possible
    final_cols = ['masfem_scaled', 'log_alldeaths', 'wind', 'min', 'category', 'year_c', 'elapsedyrs']
    for c in final_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit OLS models with robust SEs to test whether more feminine hurricane names (masfem_scaled)
    are associated with higher fatalities (proxy for fewer precautionary measures).

    Returns a dict of fitted models (statsmodels RegressionResults) and prints summaries.
    """
    results = {}

    # Ensure a copy is used
    df = df.copy()

    # Helper to prepare X and y safely: select columns, drop rows with NA, coerce numeric, add constant
    def _prepare_xy(df_local: pd.DataFrame, x_cols, y_col):
        # Keep only columns that exist
        x_cols = [c for c in x_cols if c in df_local.columns]
        if y_col not in df_local.columns or len(x_cols) == 0:
            return None, None
        X = df_local[x_cols].copy()
        y = df_local[y_col].copy()

        # Coerce to numeric
        X = X.apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(y, errors='coerce')

        # Drop rows with any NA in X or y
        mask = X.notna().all(axis=1) & y.notna()
        if mask.sum() == 0:
            return None, None
        X = X.loc[mask].astype(float)
        y = y.loc[mask].astype(float)

        X = sm.add_constant(X, has_constant='add')
        return X, y

    # 1) Main model: log_alldeaths ~ masfem_scaled + controls
    model_cols = ['masfem_scaled', 'wind', 'min', 'category', 'year_c', 'elapsedyrs']
    X, y = _prepare_xy(df, model_cols, 'log_alldeaths')
    if X is not None and y is not None:
        res_main = sm.OLS(y, X).fit(cov_type='HC3')
        print('Main model: log_alldeaths ~ masfem_scaled + controls')
        print(res_main.summary())
        results['alldeaths_masfem_scaled'] = res_main
    else:
        results['alldeaths_masfem_scaled'] = None

    # 2) Alternative IV: binary female name indicator
    alt_cols = ['gender_female', 'wind', 'min', 'category', 'year_c', 'elapsedyrs']
    X2, y2 = _prepare_xy(df, alt_cols, 'log_alldeaths')
    if X2 is not None and y2 is not None:
        res_alt = sm.OLS(y2, X2).fit(cov_type='HC3')
        print('\nAlternative IV model: log_alldeaths ~ gender_female + controls')
        print(res_alt.summary())
        results['alldeaths_gender_female'] = res_alt
    else:
        results['alldeaths_gender_female'] = None

    # 3) Robustness: use property damage as alternative DV if available
    robust_cols = ['masfem_scaled', 'wind', 'min', 'category', 'year_c', 'elapsedyrs']
    X3, y3 = _prepare_xy(df, robust_cols, 'log_ndam15')
    if X3 is not None and y3 is not None:
        res_damage = sm.OLS(y3, X3).fit(cov_type='HC3')
        print('\nRobustness model: log_ndam15 ~ masfem_scaled + controls')
        print(res_damage.summary())
        results['ndam15_masfem_scaled'] = res_damage
    else:
        results['ndam15_masfem_scaled'] = None

    return results