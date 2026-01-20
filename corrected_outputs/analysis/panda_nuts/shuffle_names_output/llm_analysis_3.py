from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/panda_nuts/shuffle_names_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and derive variables needed for modeling nut-cracking efficiency.

    Final dataframe columns used in the model:
      - NutsOpened: numeric count of nuts opened in session (from column 'nuts_opened' if present, else from 'help' if that looks like counts)
      - SessionSeconds: numeric duration of session in seconds (from column 'seconds')
      - AgeYears: numeric age in years (parsed from column 'sex' which in this dataset appears to contain numeric ages)
      - Sex: categorical sex label (from column 'age' which appears to contain 'f'/'m')
      - ReceivedHelp: binary indicator (0/1) of whether the chimpanzee received help (from 'chimpanzee')
      - HammerType: hammer type (from 'hammer')
      - NutsPerSecond: derived nuts opened / seconds
      - LogNutsPerSecond: log-transformed NutsPerSecond used as DV in the model

    Notes: the provided dataset schema has mixed/ambiguous descriptions. The transform uses sensible mappings based on column names:
      - treats 'nuts_opened' as nuts opened (fallback to 'help' if 'nuts_opened' missing)
      - treats 'seconds' as session duration in seconds
      - treats 'sex' as numeric age values and 'age' as categorical sex labels (values 'f'/'m')

    """
    # Work on a copy
    df = df.copy()

    # Standardize column names availability
    # Map nuts opened: prefer 'nuts_opened' if present and numeric; otherwise try 'help'
    if 'nuts_opened' in df.columns:
        df['NutsOpened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    else:
        df['NutsOpened'] = pd.to_numeric(df.get('help', pd.Series(dtype=float)), errors='coerce')

    # Session duration in seconds: prefer 'seconds'
    if 'seconds' in df.columns:
        df['SessionSeconds'] = pd.to_numeric(df['seconds'], errors='coerce')
    else:
        # fallback: try 'sex' or other columns, but set to NaN if not present
        df['SessionSeconds'] = pd.to_numeric(df.get('sex', pd.Series(dtype=float)), errors='coerce')

    # AgeYears: based on schema the numeric ages appear in column named 'sex' -> coerce numeric
    if 'sex' in df.columns:
        df['AgeYears'] = pd.to_numeric(df['sex'], errors='coerce')
    else:
        df['AgeYears'] = pd.Series(dtype=float)

    # Sex: based on schema the column named 'age' contains 'f'/'m' values
    if 'age' in df.columns:
        # Normalize to 'F'/'M' and treat unknowns as NaN
        df['Sex'] = df['age'].astype(str).str.strip().str.lower().map({'f': 'F', 'female': 'F', 'm': 'M', 'male': 'M'})
    else:
        df['Sex'] = pd.Series(dtype=object)

    # ReceivedHelp: map 'chimpanzee' yes/no to 1/0. Accept common variants.
    if 'chimpanzee' in df.columns:
        df['ReceivedHelp'] = df['chimpanzee'].astype(str).str.strip().str.lower().map(
            lambda x: 1 if x in ['y', 'yes', 'true', 't', '1'] else (0 if x in ['n', 'no', 'false', 'f', '0'] else np.nan)
        )
    else:
        # fallback: if there is a column called 'help' that is binary, use it (but avoid overwriting counts)
        df['ReceivedHelp'] = np.nan

    # HammerType: keep as categorical if present
    if 'hammer' in df.columns:
        df['HammerType'] = df['hammer'].astype(str).str.strip()
    else:
        df['HammerType'] = pd.Series(dtype=object)

    # If NutsOpened is mostly missing but 'help' looks like a count column, use it as NutsOpened
    if df['NutsOpened'].isna().mean() > 0.5 and 'help' in df.columns:
        # only use 'help' when it appears numeric and not a binary helper flag
        possible_counts = pd.to_numeric(df['help'], errors='coerce')
        if possible_counts.notna().sum() > 0:
            df['NutsOpened'] = possible_counts

    # Compute NutsPerSecond and drop impossible rows
    df['NutsPerSecond'] = np.nan
    mask_valid = (pd.to_numeric(df['NutsOpened'], errors='coerce').notna()) & (pd.to_numeric(df['SessionSeconds'], errors='coerce').notna())
    mask_positive = (df['SessionSeconds'] > 0)
    mask = mask_valid & mask_positive
    df.loc[mask, 'NutsPerSecond'] = df.loc[mask, 'NutsOpened'] / df.loc[mask, 'SessionSeconds']

    # Log-transform DV (add small constant to avoid log(0))
    df['LogNutsPerSecond'] = np.nan
    small = 1e-6
    df.loc[df['NutsPerSecond'].notna(), 'LogNutsPerSecond'] = np.log(df.loc[df['NutsPerSecond'].notna(), 'NutsPerSecond'] + small)

    # Keep only columns needed for modeling (and some diagnostics) in the returned dataframe
    keep_cols = ['NutsOpened', 'SessionSeconds', 'AgeYears', 'Sex', 'ReceivedHelp', 'HammerType', 'NutsPerSecond', 'LogNutsPerSecond']
    for c in keep_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Drop rows missing the dependent variable or the core independent variables
    df = df.dropna(subset=['LogNutsPerSecond', 'AgeYears', 'Sex', 'ReceivedHelp'], how='any')

    # Ensure categorical types for factor variables
    df['Sex'] = df['Sex'].astype('category')
    df['HammerType'] = df['HammerType'].astype('category')

    # Convert ReceivedHelp to a plain numpy integer dtype (not pandas nullable Int64) so patsy/statsmodels can handle it
    # After dropna above, there should be no missing values left in ReceivedHelp
    df['ReceivedHelp'] = pd.to_numeric(df['ReceivedHelp'], errors='coerce').astype(int)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS model predicting log nuts-per-second from Age, Sex, and ReceivedHelp,
    controlling for hammer type. Include interactions between ReceivedHelp and Age and Sex
    to test whether help moderates the effects of age or sex.

    Returns the fitted statsmodels regression results object.
    """
    import statsmodels.formula.api as smf

    # Ensure the transformed columns are present
    required = ['LogNutsPerSecond', 'AgeYears', 'Sex', 'ReceivedHelp', 'HammerType']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Convert ReceivedHelp to numeric 0/1 if it's not already a numeric numpy dtype
    df = df.copy()
    df['ReceivedHelp'] = pd.to_numeric(df['ReceivedHelp'], errors='coerce')

    # Build formula: main effects + interactions of ReceivedHelp with AgeYears and Sex; control for hammer type
    # Use categorical encoding for Sex and HammerType via C()
    formula = 'LogNutsPerSecond ~ AgeYears * ReceivedHelp + C(Sex) * ReceivedHelp + C(HammerType)'

    # Fit OLS with robust (HC3) standard errors
    model = smf.ols(formula, data=df).fit()

    # Attach robust covariance summary as well
    try:
        robust = model.get_robustcov_results(cov_type='HC3')
    except Exception:
        robust = model

    # Return both ordinary and robust summaries in a dict for downstream inspection
    results = {
        'ols_result': model,
        'robust_result': robust,
        'formula': formula,
        'n_obs': int(model.nobs)
    }
    return results