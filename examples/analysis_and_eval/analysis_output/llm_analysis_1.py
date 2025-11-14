from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/projects/binyu/hao_huang/stat-genie/.venv/lib/python3.11/site-packages/blade_bench/datasets/hurricane/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataframe to the analysis-ready dataframe.

    Creates standardized (z-scored) versions of the key continuous predictors,
    a log-transformed damage outcome, and a cleaned categorical source column
    for use in formula-based models. Drops rows missing required variables for
    the core analyses.
    """
    df = df.copy()

    # Ensure key columns exist
    required_cols = ['alldeaths', 'masfem', 'wind', 'min', 'category', 'ndam15', 'gender_mf', 'elapsedyrs', 'source']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for transform: {missing}")

    # Drop rows missing core variables (primary analyses need these)
    df = df.dropna(subset=['alldeaths', 'masfem', 'wind', 'min', 'category', 'ndam15', 'gender_mf', 'elapsedyrs', 'source'])

    # Create log-transformed damage (secondary DV). Add 1 to avoid log(0).
    df['log_ndam15'] = np.log(df['ndam15'].astype(float) + 1.0)

    # Standardize (z-score) continuous predictors for interpretability
    # Use population std (ddof=0) to match most standardization practices in regressions
    def zscore(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        return (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) != 0 else 1.0)

    df['masfem_z'] = zscore(df['masfem'])

    # masfem_mturk is an alternative measure; create standardized version if present
    if 'masfem_mturk' in df.columns:
        df['masfem_mturk_z'] = zscore(df['masfem_mturk'])
    else:
        # create column with NaNs so downstream code can reference it safely
        df['masfem_mturk_z'] = np.nan

    df['wind_z'] = zscore(df['wind'])
    df['min_z'] = zscore(df['min'])

    # Ensure numeric types where expected
    df['alldeaths'] = pd.to_numeric(df['alldeaths'], errors='coerce').fillna(0).astype(int)
    df['category'] = pd.to_numeric(df['category'], errors='coerce')
    df['elapsedyrs'] = pd.to_numeric(df['elapsedyrs'], errors='coerce')
    df['gender_mf'] = pd.to_numeric(df['gender_mf'], errors='coerce').fillna(0).astype(int)

    # Create a cleaned source categorical column (safe names) for use with C(source_cat) in formulas
    df['source_cat'] = df['source'].astype(str).str.replace(r'[^0-9A-Za-z]+', '_', regex=True).str.strip('_')

    # Final safety: drop any rows that still have NA in model columns
    model_cols = ['alldeaths', 'masfem_z', 'wind_z', 'min_z', 'category', 'elapsedyrs', 'gender_mf', 'log_ndam15', 'source_cat']
    df = df.dropna(subset=model_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit statistical models to test whether more feminine hurricane names are
    associated with differences in downstream consequences (used as proxies for
    fewer precautionary measures).

    Models fitted:
    1) Negative Binomial regression predicting alldeaths with masfem_z (primary IV),
       controlling for objective severity (wind_z, min_z, category), elapsedyrs,
       gender_mf, and source (categorical).
    2) Alternative Negative Binomial using masfem_mturk_z (MTurk rating) instead of masfem_z.
    3) OLS regression predicting log_ndam15 (log damage) with the same predictors.

    Returns a dict with fitted model result objects.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    results = {}

    # Primary NB model using masfem_z
    formula_nb = 'alldeaths ~ masfem_z + gender_mf + wind_z + min_z + category + elapsedyrs + C(source_cat)'
    try:
        nb_model = smf.glm(formula=formula_nb, data=df, family=sm.families.NegativeBinomial()).fit()
        results['nb_masfem'] = nb_model
    except Exception as e:
        results['nb_masfem_error'] = str(e)

    # Alternative NB model using masfem_mturk_z (drop rows missing that variable)
    if df['masfem_mturk_z'].notna().any():
        df_mturk = df.dropna(subset=['masfem_mturk_z'])
        formula_nb_mturk = 'alldeaths ~ masfem_mturk_z + gender_mf + wind_z + min_z + category + elapsedyrs + C(source_cat)'
        try:
            nb_model_mturk = smf.glm(formula=formula_nb_mturk, data=df_mturk, family=sm.families.NegativeBinomial()).fit()
            results['nb_masfem_mturk'] = nb_model_mturk
        except Exception as e:
            results['nb_masfem_mturk_error'] = str(e)
    else:
        results['nb_masfem_mturk'] = None

    # OLS on log damage (secondary outcome)
    formula_ols = 'log_ndam15 ~ masfem_z + gender_mf + wind_z + min_z + category + elapsedyrs + C(source_cat)'
    try:
        ols_model = smf.ols(formula=formula_ols, data=df).fit()
        results['ols_log_ndam15'] = ols_model
    except Exception as e:
        results['ols_log_ndam15_error'] = str(e)

    # Return results (caller can print .summary() for each fitted model)
    return results


