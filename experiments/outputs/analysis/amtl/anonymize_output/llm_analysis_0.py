from typing import Any, Dict, FrozenSet, List, Literal, Optional, Set, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for binomial modelling of AMTL.

    Produces/keeps these columns required for modeling:
      - MissingCount: integer count of missing teeth for the tooth class
      - ObservableSockets: integer count of observable sockets (trials)
      - Genus: genus string (kept for QA / potential further contrasts)
      - IsHuman: binary indicator (1 if Homo sapiens, 0 otherwise)
      - AgeAtDeath: original age estimate
      - AgeUncertainty: original uncertainty estimate (kept but not required for the primary model)
      - Age_c: centered age (AgeAtDeath - mean(AgeAtDeath)) used in model
      - SexEstimate: numeric sex estimate
      - ToothClass: categorical tooth class
      - SpecimenID, Region: retained for reference / clustering if needed
    """
    df = df.copy()

    # If input uses a schema with generic feature names, try renaming them.
    # Keep this defensive: only rename if those generic columns exist.
    rename_map = {
        'feature1': 'ToothClass',
        'feature2': 'SpecimenID',
        'feature3': 'MissingCount',
        'feature4': 'ObservableSockets',
        'feature5': 'AgeAtDeath',
        'feature6': 'AgeUncertainty',
        'feature7': 'SexEstimate',
        'feature8': 'Genus',
        'feature9': 'Region'
    }
    existing_renames = {k: v for k, v in rename_map.items() if k in df.columns}
    if existing_renames:
        df = df.rename(columns=existing_renames)

    # Drop rows missing essential fields for binomial modeling
    df = df.dropna(subset=['MissingCount', 'ObservableSockets', 'AgeAtDeath', 'SexEstimate', 'ToothClass', 'Genus'])

    # Ensure numeric types and sensible values
    df['ObservableSockets'] = pd.to_numeric(df['ObservableSockets'], errors='coerce')
    df['MissingCount'] = pd.to_numeric(df['MissingCount'], errors='coerce')
    df = df.dropna(subset=['ObservableSockets', 'MissingCount'])

    # Ensure integer counts
    # Use floor to be conservative if floats are present; then cast to int
    df['ObservableSockets'] = np.floor(df['ObservableSockets']).astype(int)
    df['MissingCount'] = np.floor(df['MissingCount']).astype(int)

    # Remove any rows with zero or negative observable sockets (cannot model binomial trials)
    df = df[df['ObservableSockets'] > 0].copy()

    # Cap MissingCount to ObservableSockets (cleaning potential data entry errors)
    df['MissingCount'] = df[['MissingCount', 'ObservableSockets']].min(axis=1).astype(int)
    df.loc[df['MissingCount'] < 0, 'MissingCount'] = 0

    # Create binary indicator for Homo sapiens (IsHuman = 1) vs others (0)
    df['Genus'] = df['Genus'].astype(str).str.strip()
    df['IsHuman'] = (df['Genus'] == 'Homo sapiens').astype(int)

    # Center AgeAtDeath for numerical stability and interpretability
    df['AgeAtDeath'] = pd.to_numeric(df['AgeAtDeath'], errors='coerce')
    df = df.dropna(subset=['AgeAtDeath'])
    df['Age_c'] = df['AgeAtDeath'] - df['AgeAtDeath'].mean()

    # Ensure SexEstimate numeric (dataset gives an estimate between 0 and 1)
    df['SexEstimate'] = pd.to_numeric(df['SexEstimate'], errors='coerce')
    df = df.dropna(subset=['SexEstimate'])

    # Ensure ToothClass is categorical and standardized
    df['ToothClass'] = df['ToothClass'].astype(str).str.strip()
    df['ToothClass'] = df['ToothClass'].astype('category')

    # Keep only the columns needed for modeling plus useful metadata
    keep_cols = [
        'SpecimenID', 'ToothClass', 'MissingCount', 'ObservableSockets',
        'AgeAtDeath', 'AgeUncertainty', 'Age_c', 'SexEstimate', 'Genus', 'Region', 'IsHuman'
    ]
    # Some datasets may not have Region/SpecimenID after cleaning; filter defensively
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial GLM for AMTL with the following specification:
      MissingCount ~ IsHuman + Age_c + SexEstimate + C(ToothClass)
    The binomial trials (number of sockets) are passed via freq_weights so the model treats
    MissingCount as the count of successes out of ObservableSockets trials.

    Returns the fitted statsmodels GLMResults object.
    """
    # Ensure required columns are present
    required = ['MissingCount', 'ObservableSockets', 'IsHuman', 'Age_c', 'SexEstimate', 'ToothClass']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Defensive checks to ensure valid values
    if (df['ObservableSockets'] <= 0).any():
        raise ValueError("ObservableSockets must be positive for all rows.")
    if (df['MissingCount'] < 0).any():
        raise ValueError("MissingCount must be non-negative for all rows.")
    if (df['MissingCount'] > df['ObservableSockets']).any():
        # Cap to be safe (should have been handled in transform, but be defensive)
        df = df.copy()
        df['MissingCount'] = df[['MissingCount', 'ObservableSockets']].min(axis=1).astype(int)

    # Build formula: use raw counts as response and provide freq_weights=ObservableSockets
    formula = 'MissingCount ~ IsHuman + Age_c + SexEstimate + C(ToothClass)'

    glm_binom = smf.glm(formula=formula,
                        data=df,
                        family=sm.families.Binomial(),
                        freq_weights=df['ObservableSockets'])

    # Attempt a normal fit; if it fails due to initial deviance nan, retry with sensible start params
    try:
        results = glm_binom.fit()
    except ValueError as e:
        # Prepare a fallback start_params using the overall proportion as intercept
        msg = str(e)
        # Only handle the specific deviance nan case here; re-raise otherwise
        if "deviance function returned a nan" not in msg:
            raise

        # Compute overall proportion of successes
        total_success = df['MissingCount'].sum()
        total_trials = df['ObservableSockets'].sum()
        if total_trials <= 0:
            raise ValueError("Total ObservableSockets must be positive to initialize model.")
        prop = total_success / total_trials
        # Clip to avoid exact 0 or 1
        eps = 1e-6
        prop = np.clip(prop, eps, 1 - eps)
        # logit intercept
        intercept = np.log(prop / (1 - prop))

        # Build start_params vector: intercept + zeros for remaining params
        # glm_binom.exog_names is available on the model instance
        try:
            n_params = len(glm_binom.exog_names)
        except Exception:
            # Fallback: try to build design matrix to get shape
            import patsy
            _, X = patsy.dmatrices(formula, data=df, return_type='dataframe')
            n_params = X.shape[1]

        start_params = np.zeros(n_params)
        start_params[0] = intercept

        results = glm_binom.fit(start_params=start_params, maxiter=200)

    print(results.summary())

    return results