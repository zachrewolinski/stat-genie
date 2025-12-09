from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Attempt to read dataset (path from original file). If running in other environments,
# users should supply their own df to transform() instead of relying on this read.
try:
    df = pd.read_csv('/accounts/campus/austin.zane/stat-genie/.venv/lib/python3.11/site-packages/blade_bench/datasets/fish/data.csv')
except Exception:
    # If file not available in the environment, leave df undefined; users should call transform with their own pd.DataFrame.
    df = None


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the analysis dataframe.

    Steps:
    - Rename input columns to clear variable names used by the model if necessary.
      If none of the target column names are present, infer mapping from the first
      six columns of the input dataframe to the required conceptual variables.
    - Ensure numeric types and drop rows with missing critical values (FishCaught, Hours).
    - Remove rows with non-positive Hours (can't be used as exposure).
    - Convert binary indicators to integer 0/1.
    - Create derived variables: GroupSize and FishPerHour.
    - Clip Hours to a small positive minimum to avoid log(0) if present.

    Returns the transformed dataframe containing at least the columns:
    ['FishCaught','Hours','LiveBait','Camper','Adults','Children','GroupSize','FishPerHour']
    """
    if df is None:
        raise ValueError("No dataframe provided to transform. Provide a pandas DataFrame as input.")

    df = df.copy()

    # Target conceptual variable column names (must not be changed)
    target_order = ['FishCaught', 'LiveBait', 'Camper', 'Adults', 'Children', 'Hours']

    # If none of the target columns are present, try to infer mapping from the first six columns.
    if not any(col in df.columns for col in target_order):
        if df.shape[1] >= 6:
            orig_cols = list(df.columns[:6])
            mapping = {orig_cols[i]: target_order[i] for i in range(6)}
            df = df.rename(columns=mapping)
        else:
            raise ValueError(
                "Input dataframe does not contain any of the required target columns "
                f"{target_order} and has fewer than 6 columns so a sensible mapping cannot be inferred."
            )

    # Now verify required columns exist
    required = ['FishCaught', 'Hours', 'LiveBait', 'Camper', 'Adults', 'Children']
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns after rename/inference: {missing_required}. "
                         f"Available columns: {list(df.columns)}")

    # Ensure numeric conversion for critical columns
    for col in required:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing critical values (count or exposure) or missing predictors
    df = df.dropna(subset=required).copy()

    # Remove observations with non-positive Hours since Hours is used as exposure/offset
    df = df[df['Hours'] > 0].copy()

    # Convert binary indicators to 0/1 integers (in case they are floats)
    # After dropna, LiveBait and Camper should be present and convertible
    df['LiveBait'] = df['LiveBait'].astype(int)
    df['Camper'] = df['Camper'].astype(int)

    # Create total group size and per-hour rate
    df['GroupSize'] = df['Adults'] + df['Children']
    df['FishPerHour'] = df['FishCaught'] / df['Hours']

    # Clip Hours to a small positive value to avoid issues computing log(Hours) as an offset
    df['Hours'] = df['Hours'].clip(lower=1e-3)

    # Final set of columns we expect for modelling
    expected_cols = ['FishCaught', 'Hours', 'LiveBait', 'Camper', 'Adults', 'Children', 'GroupSize', 'FishPerHour']
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns after transform: {missing}")

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a generalized linear model for count data using Hours as exposure (offset).

    Model specification:
    - Response: FishCaught (count)
    - Predictors: LiveBait, Camper, Adults, Children
    - Exposure/offset: log(Hours)
    - Primary family: Negative Binomial (to handle overdispersion). If fitting the Negative Binomial fails, fall back to Poisson.

    Returns the fitted model result object (statsmodels object with .summary()).
    """
    # Ensure required columns are present
    required = ['FishCaught', 'Hours', 'LiveBait', 'Camper', 'Adults', 'Children']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required for modeling: {missing}")

    # Define formula (count model with covariates). We use Adults and Children rather than GroupSize to allow distinct effects.
    formula = 'FishCaught ~ LiveBait + Camper + Adults + Children'

    # Compute offset (log of Hours) preserving alignment with df
    offset = np.log(df['Hours'])

    # Try Negative Binomial (preferred when counts are overdispersed)
    try:
        nb_model = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial(), offset=offset)
        results = nb_model.fit()
    except Exception:
        # Fallback: Poisson if Negative Binomial fitting fails
        poisson_model = smf.glm(formula=formula, data=df, family=sm.families.Poisson(), offset=offset)
        results = poisson_model.fit()

    return results