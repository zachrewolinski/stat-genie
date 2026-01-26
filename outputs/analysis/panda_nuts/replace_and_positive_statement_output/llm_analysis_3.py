from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/panda_nuts/replace_and_positive_statement_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the analysis-ready dataframe.

    Produces columns used in the models (exact names):
      - chimpanzee (unchanged)
      - age (numeric)
      - sex_male (0/1)
      - help_yes (0/1)
      - hammer (categorical, unchanged)
      - nuts_opened (count, unchanged)
      - seconds (positive numeric, unchanged)
      - nuts_per_second (numeric = nuts_opened / seconds)

    Drops rows with missing or invalid values in critical columns.
    """
    import numpy as np
    import pandas as pd

    df = df.copy()

    # Keep only the columns we need (this also tolerates extra columns)
    expected_cols = ['chimpanzee', 'age', 'sex', 'hammer', 'nuts_opened', 'seconds', 'help']
    # If any expected columns are missing, raise an informative error
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in input dataframe: {missing}")

    # Drop rows with missing critical values
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age', 'sex', 'help', 'hammer', 'chimpanzee'])

    # Ensure numeric types where expected
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age'])

    # Remove rows with non-positive session duration (cannot be used as exposure)
    df = df[df['seconds'] > 0]

    # Standardize and encode sex into a male indicator column
    # Accept common encodings like 'm', 'M', 'male', 'f', 'F', 'female'
    df['sex_str'] = df['sex'].astype(str).str.strip().str.lower()
    df['sex_male'] = df['sex_str'].isin(['m', 'male']).astype(int)
    df.drop(columns=['sex_str'], inplace=True)

    # Encode help variable into binary indicator help_yes (accept 'y','Y','yes','n','no','N')
    df['help_str'] = df['help'].astype(str).str.strip().str.lower()
    df['help_yes'] = df['help_str'].isin(['y', 'yes', 'true', '1']).astype(int)
    df.drop(columns=['help_str'], inplace=True)

    # Ensure hammer is treated as categorical (preserve original labels)
    df['hammer'] = df['hammer'].astype(str).str.strip()

    # Compute continuous efficiency for robustness checks: nuts opened per second
    # small epsilon to avoid division issues is unnecessary because seconds > 0
    df['nuts_per_second'] = df['nuts_opened'] / df['seconds']

    # Ensure chimpanzee ID is integer (used for clustering)
    try:
        df['chimpanzee'] = pd.to_numeric(df['chimpanzee'], errors='coerce').astype(int)
    except Exception:
        # If conversion fails, keep original but still usable as group label
        df['chimpanzee'] = df['chimpanzee']

    # Final sanity drop if any remaining NAs
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age', 'sex_male', 'help_yes', 'hammer', 'nuts_per_second', 'chimpanzee'])

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit models to test whether age, sex, and receiving help influence nut-cracking efficiency.

    Primary approach: Count model (Poisson with session-duration offset). Check for overdispersion
    (Pearson chi2 / df). If substantial overdispersion is present, refit with a Negative Binomial family.

    Controls: hammer as a categorical predictor. Cluster robust standard errors by chimpanzee ID.

    Secondary (robustness): OLS on nuts_per_second with clustered robust SEs.

    Returns a dictionary with the fitted model results objects and diagnostics:
      - 'final_glm_result': fitted GLMResults object (Poisson or NegBin)
      - 'glm_family': name of family used ('Poisson' or 'NegativeBinomial')
      - 'dispersion': computed Pearson chi2 / df_resid from initial Poisson
      - 'poisson_result': initial Poisson fit (results object)
      - 'ols_result': OLS results on nuts_per_second (with clustered SEs)

    Note: statsmodels result objects are returned so the caller can print .summary() as needed.
    """
    import numpy as np
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    results = {}

    # Build formula for GLM: nuts_opened ~ age + sex_male + help_yes + C(hammer)
    formula = 'nuts_opened ~ age + sex_male + help_yes + C(hammer)'

    # Fit initial Poisson with log(seconds) as offset (exposure)
    # Use statsmodels formula API for convenience with categorical hammer
    df = df.copy()
    df['log_seconds'] = np.log(df['seconds'])

    poisson_model = smf.glm(formula=formula, data=df, family=sm.families.Poisson(), offset=df['log_seconds'])
    poisson_result = poisson_model.fit()

    # Overdispersion diagnostic: Pearson chi2 / df_resid
    pearson_chi2 = sum(poisson_result.resid_pearson**2)
    df_resid = float(poisson_result.df_resid) if poisson_result.df_resid is not None else np.nan
    dispersion = (pearson_chi2 / df_resid) if df_resid and df_resid > 0 else np.nan

    results['poisson_result'] = poisson_result
    results['dispersion'] = dispersion

    # Threshold for deciding overdispersion: commonly > 1.5 indicates extra-Poisson variation
    if (not np.isnan(dispersion)) and (dispersion > 1.5):
        # Refit with a Negative Binomial family
        nb_model = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial(alpha=1.0), offset=df['log_seconds'])
        # Note: statsmodels will estimate the dispersion (alpha) if possible, but formula API's NegativeBinomial uses a parameterization; this provides a robust alternative.
        try:
            nb_result = nb_model.fit()
            final_result = nb_result
            family_used = 'NegativeBinomial'
        except Exception:
            # If NB fails, fall back to Poisson but use sandwich (clustered) SEs
            final_result = poisson_result
            family_used = 'Poisson'
    else:
        final_result = poisson_result
        family_used = 'Poisson'

    # For inference robust to within-individual correlation, compute cluster-robust covariance (by chimpanzee)
    try:
        # Get clustered covariance matrix and create a results wrapper with robust cov
        clustered = final_result.get_robustcov_results(cov_type='cluster', groups=df['chimpanzee'])
        results['final_glm_result'] = clustered
    except Exception:
        # If clustering fails for NB model or other reason, return the plain result
        results['final_glm_result'] = final_result

    results['glm_family'] = family_used

    # Robustness: OLS on nuts_per_second (continuous), with clustered SEs by chimpanzee
    ols_formula = 'nuts_per_second ~ age + sex_male + help_yes + C(hammer)'
    ols_model = smf.ols(formula=ols_formula, data=df)
    ols_result = ols_model.fit()

    try:
        ols_clustered = ols_result.get_robustcov_results(cov_type='cluster', groups=df['chimpanzee'])
        results['ols_result'] = ols_clustered
    except Exception:
        results['ols_result'] = ols_result

    # Add model summaries as strings for quick inspection
    try:
        results['poisson_summary'] = poisson_result.summary().as_text()
    except Exception:
        results['poisson_summary'] = None
    try:
        results['final_glm_summary'] = results['final_glm_result'].summary().as_text()
    except Exception:
        results['final_glm_summary'] = None
    try:
        results['ols_summary'] = results['ols_result'].summary().as_text()
    except Exception:
        results['ols_summary'] = None

    return results


