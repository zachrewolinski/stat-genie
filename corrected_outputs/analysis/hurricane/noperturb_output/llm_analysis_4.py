from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/hurricane/noperturb_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataframe into analysis-ready form.
    Produces the following new columns used in modeling:
      - log_alldeaths: log(alldeaths + 1)
      - masfem_z: standardized masfem (mean 0, sd 1)
      - masfem_mturk_z: standardized masfem_mturk (if present)
      - female_name: integer version of gender_mf (0/1)
      - ndam15_log: log(ndam15 + 1) if ndam15 present
      - year_centered: year minus mean(year)
      - ensures category and source are categorical
    """
    # Work on a copy
    df = df.copy()

    # Required columns for primary analyses: alldeaths and masfem
    # Drop rows missing either of these; other missingness will be handled before model-fitting
    df = df.dropna(subset=['alldeaths', 'masfem'])

    # Dependent variable: log-transform deaths to reduce skew (add 1 to keep zeros)
    df['log_alldeaths'] = np.log(df['alldeaths'].astype(float) + 1.0)

    # Independent variables: standardize continuous femininity score(s)
    df['masfem_z'] = (df['masfem'].astype(float) - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0)

    # MTurk measure may be absent in some versions of the dataset; create standardized version only if present
    if 'masfem_mturk' in df.columns:
        df['masfem_mturk_z'] = (df['masfem_mturk'].astype(float) - df['masfem_mturk'].mean()) / (df['masfem_mturk'].std(ddof=0) if df['masfem_mturk'].std(ddof=0) != 0 else 1.0)
    else:
        # create column of NaNs to keep column list consistent
        df['masfem_mturk_z'] = np.nan

    # Binary female-coded name indicator from provided column
    if 'gender_mf' in df.columns:
        df['female_name'] = df['gender_mf'].astype(int)
    else:
        df['female_name'] = np.nan

    # Logged damage (ndam15) to control for economic magnitude (if available)
    if 'ndam15' in df.columns:
        df['ndam15_log'] = np.log(df['ndam15'].astype(float) + 1.0)
    else:
        df['ndam15_log'] = np.nan

    # Center year to help interpretation and reduce collinearity
    if 'year' in df.columns:
        df['year_centered'] = df['year'].astype(float) - df['year'].astype(float).mean()
    else:
        df['year_centered'] = np.nan

    # Ensure categorical columns are typed as category for modeling routines that respect C(...)
    if 'category' in df.columns:
        df['category'] = df['category'].astype('category')

    if 'source' in df.columns:
        df['source'] = df['source'].astype('category')

    # Keep the transformed dataframe (do not drop additional rows here; model code will handle any remaining NA in model variables)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit primary and robustness models testing whether femininity of hurricane names predicts fatalities,
    controlling for storm intensity and other covariates.

    Returns a dictionary with keys:
      - 'ols': OLS results object on log(alldeaths + 1)
      - 'negbin': Negative binomial GLM results object on alldeaths (count model)

    Primary specification: log_alldeaths ~ masfem_z + wind + min + ndam15_log + year_centered + elapsedyrs + C(category) + C(source)
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import patsy

    # Prepare dataframe: drop rows with missing values in any columns used by the primary formula
    formula_continuous = 'log_alldeaths ~ masfem_z + wind + min + ndam15_log + year_centered + elapsedyrs + C(category) + C(source)'
    required_vars = ['log_alldeaths', 'masfem_z', 'wind', 'min', 'ndam15_log', 'year_centered', 'elapsedyrs', 'category', 'source']
    df_model = df.dropna(subset=required_vars).copy()

    # 1) OLS on log deaths (primary): interpretable as percent-like changes on multiplicative scale
    ols_res = smf.ols(formula_continuous, data=df_model).fit()

    # 2) Negative binomial on counts (robustness): model raw counts with overdispersion
    # Build design matrices for count model using patsy to ensure same dummies/encoding as formula
    # Use alldeaths (raw counts) as response
    try:
        y_nb, X_nb = patsy.dmatrices('alldeaths ~ masfem_z + wind + min + ndam15_log + year_centered + elapsedyrs + C(category) + C(source)',
                                    data=df_model, return_type='dataframe')
        # Convert y to a 1d array of counts
        y_nb = np.asarray(y_nb).ravel()
        # Fit GLM Negative Binomial
        negbin_model = sm.GLM(y_nb, X_nb, family=sm.families.NegativeBinomial())
        negbin_res = negbin_model.fit()
    except Exception as e:
        negbin_res = None

    # Return both fitted results for inspection
    return {
        'ols': ols_res,
        'negbin': negbin_res
    }


