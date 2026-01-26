from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/noperturb_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataframe into the modeling dataframe.

    Produces the following model-ready columns (all used in the modeling function):
      - alldeaths: original death counts (numeric)
      - log_deaths: log1p(alldeaths)
      - masfem: original masfem score (continuous)
      - masfem_z: standardized masfem (mean 0, sd 1)
      - gender_female: integer 0/1 from gender_mf (alternative IV)
      - wind, min, category, elapsedyrs, year: numeric controls
      - ndam15, log_ndam15: damage; log transformed

    Notes: drop rows with missing essential variables for modeling.
    """
    df = df.copy()

    # Ensure numeric columns are numeric where possible
    numeric_cols = ['alldeaths', 'masfem', 'masfem_mturk', 'gender_mf', 'wind', 'min', 'category', 'elapsedyrs', 'ndam15', 'year']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing the essential columns for the main analysis
    required_for_model = [c for c in ['alldeaths', 'masfem', 'wind', 'category'] if c in df.columns]
    df = df.dropna(subset=required_for_model)

    # Dependent variable: log(1 + alldeaths)
    df['alldeaths'] = df['alldeaths'].astype(float)
    df['log_deaths'] = np.log1p(df['alldeaths'])

    # Independent variable: continuous masfem and its standardized version
    df['masfem'] = df['masfem'].astype(float)
    # Use population std (ddof=0) to mirror common z-scoring for predictors
    mas_mean = df['masfem'].mean()
    mas_std = df['masfem'].std(ddof=0)
    if mas_std == 0 or np.isnan(mas_std):
        # fallback to sample std if population std is zero
        mas_std = df['masfem'].std(ddof=1)
    df['masfem_z'] = (df['masfem'] - mas_mean) / (mas_std if mas_std != 0 else 1.0)

    # Alternative / robustness IV: binary female-coded name
    if 'gender_mf' in df.columns:
        # gender_mf is 0 (male) or 1 (female) per schema
        df['gender_female'] = df['gender_mf'].astype('Int64')
    else:
        df['gender_female'] = pd.Series([pd.NA] * len(df), index=df.index, dtype='Int64')

    # Controls: coerce to appropriate types and fill small missingness where sensible
    if 'ndam15' in df.columns:
        df['ndam15'] = pd.to_numeric(df['ndam15'], errors='coerce')
        df['log_ndam15'] = np.log1p(df['ndam15'].fillna(0.0))
    else:
        df['ndam15'] = pd.NA
        df['log_ndam15'] = np.nan

    # Ensure category and wind and min are numeric
    if 'category' in df.columns:
        df['category'] = pd.to_numeric(df['category'], errors='coerce').astype(pd.Int64Dtype())
    if 'wind' in df.columns:
        df['wind'] = pd.to_numeric(df['wind'], errors='coerce')
    if 'min' in df.columns:
        df['min'] = pd.to_numeric(df['min'], errors='coerce')

    # Ensure elapsedyrs present (if not, create from year relative to max year as a fallback)
    if 'elapsedyrs' not in df.columns or df['elapsedyrs'].isna().all():
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            # Use difference from max year in dataset as a proxy for elapsedyrs
            max_year = int(df['year'].max()) if not df['year'].isna().all() else pd.NA
            if pd.notna(max_year):
                df['elapsedyrs'] = max_year - df['year']
            else:
                df['elapsedyrs'] = pd.NA
        else:
            df['elapsedyrs'] = pd.NA

    # Final drop: ensure no missingness in the essential columns used by the model
    final_model_cols = ['log_deaths', 'masfem_z', 'wind', 'min', 'category', 'elapsedyrs', 'log_ndam15', 'gender_female']
    # Keep only columns that actually exist in df
    final_existing = [c for c in final_model_cols if c in df.columns]
    df = df.dropna(subset=[c for c in ['log_deaths', 'masfem_z', 'wind', 'category'] if c in df.columns])

    # Return transformed dataframe containing at least the columns used in modeling
    return df

# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Run main statistical models to test whether more-feminine hurricane names (masfem_z)
    are associated with higher fatalities (interpreted as less precaution), controlling
    for storm severity and other covariates.

    Returns a dictionary with statsmodels results objects for:
      - OLS on log(1 + deaths) with robust (HC3) standard errors (primary continuous-outcome specification)
      - Poisson GLM on raw death counts with robust SE (count model robustness check)
      - Negative binomial GLM (if it fits) as an additional robustness check for overdispersion
    """
    results = {}
    df = df.copy()

    # Define outcome and covariates: include only columns present
    y_ols = df['log_deaths']
    # Build X with prespecified controls; keep only columns that exist
    candidate_X = ['masfem_z', 'gender_female', 'wind', 'min', 'category', 'elapsedyrs', 'log_ndam15']
    X_cols = [c for c in candidate_X if c in df.columns]

    X = df[X_cols].copy()
    # For safety, convert integer-dtype categories to numeric floats for modeling
    for c in X.columns:
        if pd.api.types.is_integer_dtype(X[c].dtype) or pd.api.types.is_bool_dtype(X[c].dtype):
            X[c] = pd.to_numeric(X[c], errors='coerce')

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # 1) OLS on log deaths (primary specification)
    try:
        ols_model = sm.OLS(y_ols, X).fit()
        # attach heteroskedasticity-robust (HC3) covariance
        ols_robust = ols_model.get_robustcov_results(cov_type='HC3')
        results['ols_robust'] = ols_robust
    except Exception as e:
        results['ols_error'] = str(e)

    # 2) Poisson GLM on counts (robustness for count nature of outcome)
    if 'alldeaths' in df.columns:
        y_count = df['alldeaths'].astype(float)
        try:
            poisson = sm.GLM(y_count, X, family=sm.families.Poisson()).fit()
            poisson_robust = poisson.get_robustcov_results(cov_type='HC3')
            results['poisson_robust'] = poisson_robust
        except Exception as e:
            results['poisson_error'] = str(e)

        # 3) Negative binomial GLM as another robustness check (handles overdispersion)
        try:
            nb = sm.GLM(y_count, X, family=sm.families.NegativeBinomial()).fit()
            nb_robust = nb.get_robustcov_results(cov_type='HC3')
            results['negbin_robust'] = nb_robust
        except Exception as e:
            # some versions/environments may not converge or support this family; capture error
            results['negbin_error'] = str(e)

    # Provide a brief textual guidance about interpreting the results in the return dict
    results['interpretation_note'] = (
        "Primary inference: coefficient on 'masfem_z' tests whether more-feminine names are associated "
        "with higher log(1+deaths) (positive coefficient -> more deaths -> consistent with less precaution), "
        "conditional on the included severity controls. Check poisson/negbin for count-model robustness. "
        "All models include the controls present in the dataframe: %s" % (', '.join(X_cols))
    )

    return results

