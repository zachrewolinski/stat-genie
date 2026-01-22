from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm


df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/add_features_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataframe into analysis-ready dataframe.

    Produces standardized femininity score(s), log outcomes, a severity index, and basic cleaning.

    Returns a dataframe that contains the exact column names required by the
    analysis contract (where possible). Intermediate helper columns may be
    created but are not exposed beyond this transform.
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Ensure numeric columns are numeric where appropriate
    numeric_cols = [
        'masfem', 'masfem_mturk', 'alldeaths', 'ndam15',
        'wind', 'min', 'category', 'year', 'elapsedyrs'
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Handle gender_mf robustly to produce name_female (binary).
    # If gender_mf exists, attempt to map common encodings to 0/1.
    # If it does not exist, create a name_female column defaulting to 0
    # (so the column is present in the final dataframe as required).
    def _map_gender(x):
        if pd.isna(x):
            return np.nan
        # numeric-like
        if isinstance(x, (int, float, np.integer, np.floating)) and not np.isnan(x):
            try:
                xi = int(x)
                return 1 if xi == 1 else 0
            except Exception:
                pass
        s = str(x).strip().lower()
        if s in ('female', 'f', '1', 'true', 't', 'yes', 'y'):
            return 1
        if s in ('male', 'm', '0', 'false', 'fa', 'no', 'n'):
            return 0
        # fallback by first letter
        if s.startswith('f'):
            return 1
        if s.startswith('m'):
            return 0
        return np.nan

    if 'gender_mf' in df.columns:
        df['name_female'] = df['gender_mf'].apply(_map_gender)
        # If after mapping there are NaNs, replace those with 0 (conservative default)
        df['name_female'] = df['name_female'].fillna(0).astype(int)
    else:
        df['name_female'] = 0

    # Drop rows missing primary IV (masfem) or essential severity components if they exist.
    # We require masfem to compute z_masfem; if masfem missing drop those rows.
    required_for_masfem = []
    if 'masfem' in df.columns:
        required_for_masfem.append('masfem')
    # If severity components are present in the raw data, we require them to build the severity index.
    for c in ['wind', 'min', 'category']:
        if c in df.columns:
            required_for_masfem.append(c)
    if len(required_for_masfem) > 0:
        df = df.dropna(subset=required_for_masfem)

    # Log-transform outcomes (add 1 to avoid log(0)) if present
    if 'alldeaths' in df.columns:
        df['alldeaths'] = pd.to_numeric(df['alldeaths'], errors='coerce')
        df['log_alldeaths'] = np.log(df['alldeaths'].fillna(0) + 1)
    if 'ndam15' in df.columns:
        df['ndam15'] = pd.to_numeric(df['ndam15'], errors='coerce')
        df['log_ndam15'] = np.log(df['ndam15'].fillna(0) + 1)

    # Standardize masfem and masfem_mturk (z-scores) safely
    if 'masfem' in df.columns:
        mas_mean = df['masfem'].mean()
        mas_std = df['masfem'].std(ddof=0)
        if pd.isna(mas_std) or mas_std == 0:
            mas_std = 1.0
        df['z_masfem'] = (df['masfem'] - mas_mean) / mas_std

    if 'masfem_mturk' in df.columns:
        m_mean = df['masfem_mturk'].mean()
        m_std = df['masfem_mturk'].std(ddof=0)
        if pd.isna(m_std) or m_std == 0:
            m_std = 1.0
        df['z_masfem_mturk'] = (df['masfem_mturk'] - m_mean) / m_std

    # Standardize severity components (wind, category, min)
    for comp in ['wind', 'category', 'min']:
        if comp in df.columns:
            comp_mean = df[comp].mean()
            comp_std = df[comp].std(ddof=0)
            if pd.isna(comp_std) or comp_std == 0:
                comp_std = 1.0
            df[f'z_{comp}'] = (df[comp] - comp_mean) / comp_std

    # Build severity index: wind (higher worse), category (higher worse), min (lower worse so invert)
    comps_present = [c for c in ['z_wind', 'z_category', 'z_min'] if c in df.columns]
    if len(comps_present) >= 1:
        arr = []
        for c in comps_present:
            if c == 'z_min':
                arr.append(-1.0 * df[c])
            else:
                arr.append(df[c])
        # Sum component-wise; if only one component present this becomes that component
        df['severity_raw'] = np.sum(arr, axis=0)
        sr_mean = df['severity_raw'].mean()
        sr_std = df['severity_raw'].std(ddof=0)
        if pd.isna(sr_std) or sr_std == 0:
            sr_std = 1.0
        df['severity_z'] = (df['severity_raw'] - sr_mean) / sr_std

    # Ensure year and elapsedyrs are numeric (already coerced above), fill or drop as needed.
    # For the purposes of modeling, year and elapsedyrs must be finite numbers.
    # We'll not invent values; rows missing these will be dropped below.

    # Keep only the columns useful for analysis (but preserve original numeric columns too)
    keep_cols = []
    for c in [
        'ind', 'year', 'name', 'masfem', 'z_masfem', 'masfem_mturk', 'z_masfem_mturk',
        'gender_mf', 'name_female', 'alldeaths', 'log_alldeaths', 'ndam15', 'log_ndam15',
        'wind', 'category', 'min', 'severity_z', 'elapsedyrs'
    ]:
        if c in df.columns:
            keep_cols.append(c)

    df = df[keep_cols].reset_index(drop=True)

    # Final critical cleaning: remove rows that would cause missing data in the model.
    # The modeling code requires the following final dataframe columns:
    must_have = ['z_masfem', 'severity_z', 'wind', 'category', 'min', 'year', 'elapsedyrs', 'alldeaths']
    # Only require those that exist in the df (to avoid dropping everything if original lacked a column)
    must_have_present = [c for c in must_have if c in df.columns]
    if len(must_have_present) > 0:
        # Replace infs with NaN and drop rows with NaN in any must-have column
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=must_have_present).reset_index(drop=True)

    # Ensure name_female column exists in final dataframe (as required by the conceptual variables)
    if 'name_female' not in df.columns:
        df['name_female'] = 0

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a set of models testing the association between feminine hurricane names and (proxy) public precaution outcomes.

    Models returned:
      - OLS on log_alldeaths (robust HC3 SEs)
      - OLS on log_ndam15 (robust HC3 SEs)
      - Negative binomial (GLM) on raw alldeaths counts

    Returns a dict with fitted results objects.
    """
    results = {}

    # Require that essential transformed columns exist in the dataframe
    required = ['z_masfem', 'severity_z', 'wind', 'category', 'min', 'year', 'elapsedyrs']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Required column for modeling missing: {c}")

    # Build common control list
    controls = ['severity_z', 'wind', 'category', 'min', 'year', 'elapsedyrs']

    # Include binary female name as a covariate as well (if present)
    if 'name_female' in df.columns:
        controls_with_binary = controls + ['name_female']
    else:
        controls_with_binary = controls

    # Prepare base design matrix (we will add z_masfem per-model and clean missing data per model)
    base_X = df[controls_with_binary].copy()
    base_X = sm.add_constant(base_X, has_constant='add')

    # Helper to clean X and y for modeling: remove rows with NA/inf in either
    def _clean_xy(X: pd.DataFrame, y: pd.Series):
        # Replace inf with NaN first
        X = X.replace([np.inf, -np.inf], np.nan)
        y = y.replace([np.inf, -np.inf], np.nan)
        # Align indices and drop any rows with NaN
        combined = pd.concat([y, X], axis=1)
        combined = combined.dropna(axis=0)
        if combined.shape[0] == 0:
            raise ValueError("No observations remaining after dropping rows with missing data for model.")
        y_clean = combined.iloc[:, 0]
        X_clean = combined.iloc[:, 1:]
        # Ensure constant exists
        if 'const' not in X_clean.columns:
            X_clean = sm.add_constant(X_clean, has_constant='add')
        return X_clean, y_clean

    # 1) OLS on log_alldeaths
    if 'log_alldeaths' in df.columns:
        y = df['log_alldeaths']
        X1 = base_X.copy()
        X1['z_masfem'] = df['z_masfem']
        try:
            X1_clean, y_clean = _clean_xy(X1, y)
            ols_model = sm.OLS(y_clean, X1_clean).fit()
            # Convert to HC3 robust covariance results
            ols_deaths = ols_model.get_robustcov_results(cov_type='HC3')
            results['ols_log_alldeaths'] = ols_deaths
        except Exception as e:
            # Provide informative error if no data / other failure, but keep function robust
            raise RuntimeError(f"Failed to fit OLS on log_alldeaths: {e}")

    # 2) OLS on log_ndam15
    if 'log_ndam15' in df.columns:
        y2 = df['log_ndam15']
        X2 = base_X.copy()
        X2['z_masfem'] = df['z_masfem']
        try:
            X2_clean, y2_clean = _clean_xy(X2, y2)
            ols_model2 = sm.OLS(y2_clean, X2_clean).fit()
            ols_damage = ols_model2.get_robustcov_results(cov_type='HC3')
            results['ols_log_ndam15'] = ols_damage
        except Exception as e:
            raise RuntimeError(f"Failed to fit OLS on log_ndam15: {e}")

    # 3) Negative binomial on raw alldeaths (counts)
    if 'alldeaths' in df.columns:
        counts = df['alldeaths'].fillna(0)
        # Ensure integer counts
        try:
            counts = counts.round().astype(int)
        except Exception:
            counts = pd.to_numeric(counts, errors='coerce').fillna(0).round().astype(int)
        X3 = base_X.copy()
        X3['z_masfem'] = df['z_masfem']
        try:
            X3_clean, counts_clean = _clean_xy(X3, counts)
            # Use GLM Negative Binomial to allow overdispersion
            try:
                nb_model = sm.GLM(counts_clean, X3_clean, family=sm.families.NegativeBinomial()).fit()
                results['nb_alldeaths'] = nb_model
            except Exception:
                # fallback to Poisson if NegativeBinomial fails to converge
                pois_model = sm.GLM(counts_clean, X3_clean, family=sm.families.Poisson()).fit()
                results['pois_alldeaths'] = pois_model
        except Exception as e:
            raise RuntimeError(f"Failed to fit count model on alldeaths: {e}")

    return results