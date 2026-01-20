from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/hurricane/replace_with_rvs_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataframe into the analysis-ready dataframe.

    Produces columns used in modeling:
    - masfem_std: standardized masfem (mean 0, sd 1)
    - min_pressure: copy of original 'min' column renamed to avoid reserved-word conflicts
    - log_alldeaths: log(1 + alldeaths)
    - log_ndam15: log(1 + ndam15) (kept for exploratory checks)
    - year_centered: year centered on its mean

    Drops rows missing key variables used in the main models.
    """
    # Work on a copy
    df = df.copy()

    # Ensure numeric types where expected
    numeric_cols = ['masfem', 'gender_mf', 'category', 'alldeaths', 'ndam15', 'wind', 'min', 'elapsedyrs', 'year']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Rename pressure column to avoid potential name conflicts
    if 'min' in df.columns:
        df['min_pressure'] = df['min']

    # Create dependent variable: log(1 + alldeaths)
    # Keep original count as well
    if 'alldeaths' in df.columns:
        df['alldeaths'] = pd.to_numeric(df['alldeaths'], errors='coerce')
        df['log_alldeaths'] = np.log1p(df['alldeaths'])
    else:
        df['log_alldeaths'] = np.nan

    # Alternate outcome for damage (exploratory)
    if 'ndam15' in df.columns:
        df['ndam15'] = pd.to_numeric(df['ndam15'], errors='coerce')
        df['log_ndam15'] = np.log1p(df['ndam15'])

    # Standardize masfem (name femininity index)
    if 'masfem' in df.columns:
        df['masfem'] = pd.to_numeric(df['masfem'], errors='coerce')
        mean_m = df['masfem'].mean(skipna=True)
        std_m = df['masfem'].std(skipna=True)
        if pd.notnull(std_m) and std_m > 0:
            df['masfem_std'] = (df['masfem'] - mean_m) / std_m
        else:
            df['masfem_std'] = np.nan

    # Binary female name indicator from gender_mf (0 male, 1 female). Keep as int.
    if 'gender_mf' in df.columns:
        df['gender_female'] = df['gender_mf'].astype('float').round(0).astype('Int64')

    # Category as categorical variable (keep numeric column too)
    if 'category' in df.columns:
        df['category'] = pd.to_numeric(df['category'], errors='coerce')

    # Year centered
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['year_centered'] = df['year'] - df['year'].mean()

    # elapsedyrs numeric
    if 'elapsedyrs' in df.columns:
        df['elapsedyrs'] = pd.to_numeric(df['elapsedyrs'], errors='coerce')

    # Keep only rows with non-missing values for the DV, IV, and essential controls
    required = ['log_alldeaths', 'masfem_std', 'category', 'wind', 'min_pressure', 'year_centered', 'elapsedyrs']
    present_required = [c for c in required if c in df.columns]
    df = df.dropna(subset=present_required)

    # Final: reset index
    df = df.reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Run statistical models to test whether more feminine hurricane names are associated with higher fatalities,
    after controlling for storm intensity and temporal trends.

    Returns a dictionary with:
    - ols: OLS regression of log(1+alldeaths) on masfem_std and controls (robust HC3 SEs)
    - nb: GLM with Negative Binomial family on raw alldeaths (counts) with same predictors

    Notes:
    - OLS on log(1+alldeaths) is the primary model (handles zeros and skewness).
    - Negative binomial provides a count-based robustness check for overdispersion.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    df = df.copy()

    # Ensure required columns exist
    needed = ['log_alldeaths', 'alldeaths', 'masfem_std', 'category', 'wind', 'min_pressure', 'year_centered', 'elapsedyrs']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # OLS model on log deaths (primary)
    ols_formula = 'log_alldeaths ~ masfem_std + C(category) + wind + min_pressure + year_centered + elapsedyrs'
    ols_model = smf.ols(ols_formula, data=df).fit(cov_type='HC3')

    # Negative binomial (count model) on raw alldeaths as robustness check
    # Use GLM with NegativeBinomial family
    # Add a small constant to alldeaths? Not necessary; NB handles zeros.
    nb_formula = 'alldeaths ~ masfem_std + C(category) + wind + min_pressure + year_centered + elapsedyrs'
    try:
        nb_model = smf.glm(nb_formula, data=df, family=sm.families.NegativeBinomial()).fit()
    except Exception as e:
        nb_model = None

    # Package results
    results = {
        'ols': ols_model,
        'neg_binomial_glm': nb_model,
        'formula_ols': ols_formula,
        'formula_nb': nb_formula,
        'n_obs': int(df.shape[0])
    }

    return results


