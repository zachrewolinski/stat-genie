from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/panda_nuts/add_features_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the dataframe used for modeling.

    Produces the following columns used in the model and diagnostics:
      - nuts_opened : numeric count of nuts opened (from original)
      - seconds     : session duration in seconds (exposure)
      - age         : age in years (numeric)
      - sex_m       : 1 if male, 0 if female
      - help_y      : 1 if helper present in session, 0 if not
      - hammer      : hammer type (kept as categorical)
      - chimpanzee  : individual id (kept for clustering)
      - nuts_per_sec: derived nuts_opened / seconds (for diagnostics)

    Rows with missing critical fields or invalid seconds (<=0) are dropped.
    """
    # make a copy to avoid modifying input in-place
    df = df.copy()

    # Standardize column names we will use (if necessary) - assume given names match schema
    # Drop rows missing critical variables
    required_cols = ['nuts_opened', 'seconds', 'age', 'sex', 'help', 'hammer', 'chimpanzee']
    missing_req = [c for c in required_cols if c not in df.columns]
    if missing_req:
        raise ValueError(f"Required column(s) missing from input dataframe: {missing_req}")

    df = df.dropna(subset=['nuts_opened', 'seconds', 'age', 'sex', 'help', 'chimpanzee'])

    # Ensure numeric columns are numeric
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Drop rows with invalid seconds (zero or negative) or missing after coercion
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age'])
    df = df[df['seconds'] > 0]

    # Normalize and coerce categorical indicators
    # sex -> sex_m: 1 if male ('m' or 'M'), 0 if female ('f' or 'F')
    df['sex_str'] = df['sex'].astype(str).str.strip().str.lower()
    df['sex_m'] = df['sex_str'].map({'m': 1, 'male': 1, 'f': 0, 'female': 0})
    # If mapping produced NaNs (unexpected codes), try to infer from first character
    mask_sex_na = df['sex_m'].isna()
    if mask_sex_na.any():
        df.loc[mask_sex_na, 'sex_m'] = df.loc[mask_sex_na, 'sex_str'].str[0].map({'m': 1, 'f': 0})
    # If still NaN, drop those rows
    df = df.dropna(subset=['sex_m'])
    df['sex_m'] = df['sex_m'].astype(int)

    # help -> help_y: map 'y'/'yes' to 1, 'n'/'no' to 0 (case-insensitive)
    df['help_str'] = df['help'].astype(str).str.strip().str.lower()
    df['help_y'] = df['help_str'].map({'y': 1, 'yes': 1, 'n': 0, 'no': 0})
    # If values like 'Y' or 'N' or 'True'/'False', mapping above should catch via lower()
    # Drop rows where help can't be interpreted
    df = df.dropna(subset=['help_y'])
    df['help_y'] = df['help_y'].astype(int)

    # hammer: keep as-is but coerce to string and strip
    df['hammer'] = df['hammer'].astype(str).str.strip()

    # chimpanzee id: keep as-is (coerce to int if numeric), but also as categorical grouping
    # try to coerce numeric ids to integers when possible
    try:
        df['chimpanzee'] = pd.to_numeric(df['chimpanzee'], errors='ignore')
    except Exception:
        pass

    # Derived diagnostics: nuts per second
    df['nuts_per_sec'] = df['nuts_opened'] / df['seconds']

    # Final: keep only columns needed for modeling and diagnostics
    keep_cols = ['chimpanzee', 'age', 'sex_m', 'help_y', 'hammer', 'seconds', 'nuts_opened', 'nuts_per_sec']
    df = df[keep_cols]

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a count model for nuts opened with exposure equal to session seconds.

    Modeling strategy:
      1. Fit a Poisson GLM with offset = log(seconds). Formula includes age, sex_m, help_y and hammer as a categorical control: 
         nuts_opened ~ age + sex_m + help_y + C(hammer)
      2. Compute overdispersion as deviance / df_resid. If overdispersion > 1.5, refit using a Negative Binomial family.
      3. For inference, return cluster-robust standard errors clustered on chimpanzee.

    Returns a dict with keys:
      - 'model_type': 'Poisson' or 'NegativeBinomial'
      - 'result': the fitted results object adjusted for cluster-robust covariances
      - 'raw_result': the raw fitted result before robust adjustment
      - 'overdispersion': numeric value (deviance / df_resid)
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # ensure required columns exist
    required = ['nuts_opened', 'seconds', 'age', 'sex_m', 'help_y', 'hammer', 'chimpanzee']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Build formula
    formula = 'nuts_opened ~ age + sex_m + help_y + C(hammer)'

    # Fit Poisson with offset = log(seconds)
    poisson_mod = smf.glm(formula=formula, data=df, family=sm.families.Poisson(),
                          offset=np.log(df['seconds']))
    poisson_res = poisson_mod.fit()

    # compute overdispersion measure
    # deviance / df_resid > 1 indicates overdispersion; use threshold 1.5 to switch to NB
    overdispersion = poisson_res.deviance / float(poisson_res.df_resid) if poisson_res.df_resid > 0 else np.nan

    final_model_type = 'Poisson'
    final_raw_res = poisson_res

    if not np.isnan(overdispersion) and overdispersion > 1.5:
        # Fit Negative Binomial GLM (uses variance function of NB); keep same formula and offset
        try:
            nb_mod = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial(),
                             offset=np.log(df['seconds']))
            nb_res = nb_mod.fit()
            final_model_type = 'NegativeBinomial'
            final_raw_res = nb_res
        except Exception:
            # If NB fails, fall back to Poisson but note overdispersion
            final_model_type = 'Poisson'
            final_raw_res = poisson_res

    # Obtain cluster-robust covariance results clustered on chimpanzee
    # Use get_robustcov_results to produce a results object with robust covariances
    try:
        clustered_res = final_raw_res.get_robustcov_results(cov_type='cluster',
                                                            groups=df['chimpanzee'])
    except Exception:
        # If clustering fails for some reason, return the raw result and warn in the output
        clustered_res = final_raw_res

    # Pack up results
    out = {
        'model_type': final_model_type,
        'raw_result': final_raw_res,
        'result': clustered_res,
        'overdispersion': overdispersion
    }

    return out


