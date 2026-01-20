from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/amtl/anonymize_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for binomial GLM modeling.

    Inputs (original columns):
      - feature1: tooth class ('Anterior', 'Posterior', 'Premolar')
      - feature2: specimen id
      - feature3: number of teeth missing of given class
      - feature4: number of observable sockets that could be scored for missing teeth
      - feature5: estimated age at death
      - feature6: assigned uncertainty of age at death (kept but not used directly)
      - feature7: estimate of sex of specimen (numeric estimate between 0 and 1)
      - feature8: specimen genus (e.g., 'Homo sapiens', 'Pan', 'Pongo', 'Papio')
      - feature9: region of origin (kept but not used by default)

    Outputs (columns used in model):
      - SpecimenID, ToothClass, n_missing, n_observed, n_present,
        IsHuman, Age_c (standardized age), Sex_c (centered sex estimate)
    """
    # Work on a copy
    df = df.copy()

    # Rename raw columns to meaningful names
    df = df.rename(columns={
        'feature1': 'ToothClass',
        'feature2': 'SpecimenID',
        'feature3': 'n_missing',
        'feature4': 'n_observed',
        'feature5': 'Age',
        'feature6': 'AgeUncertainty',
        'feature7': 'SexEstimate',
        'feature8': 'Genus',
        'feature9': 'Region'
    })

    # Drop rows missing essential fields
    essential_cols = ['ToothClass', 'SpecimenID', 'n_missing', 'n_observed', 'Age', 'SexEstimate', 'Genus']
    df = df.dropna(subset=essential_cols)

    # Ensure counts are integers and sensible
    # If counts are floats due to import, round to nearest integer; then enforce non-negative
    df['n_missing'] = (df['n_missing'].round().astype(int)).clip(lower=0)
    df['n_observed'] = (df['n_observed'].round().astype(int)).clip(lower=0)

    # Remove impossible rows: no observed sockets, or missing > observed
    df = df[df['n_observed'] > 0]
    df = df[df['n_missing'] <= df['n_observed']]

    # Create n_present (failures) for binomial model
    df['n_present'] = df['n_observed'] - df['n_missing']

    # Create binary indicator for modern humans vs non-human primates
    df['IsHuman'] = (df['Genus'].astype(str).str.strip() == 'Homo sapiens').astype(int)

    # Normalize / center numerical covariates
    # Standardize age to mean 0, sd 1 (z-score) to aid interpretability and modeling
    df['Age_c'] = (df['Age'] - df['Age'].mean()) / (df['Age'].std(ddof=0) if df['Age'].std(ddof=0) != 0 else 1.0)

    # Center sex estimate (keep as continuous). If the variable encodes probability of female, centering is appropriate.
    df['Sex_c'] = df['SexEstimate'] - df['SexEstimate'].mean()

    # Ensure ToothClass is categorical and standardized strings
    df['ToothClass'] = df['ToothClass'].astype(str).str.strip().str.capitalize()
    df['ToothClass'] = pd.Categorical(df['ToothClass'], categories=['Anterior', 'Posterior', 'Premolar'])

    # SpecimenID should be treated as categorical for clustering but keep its original values
    df['SpecimenID'] = df['SpecimenID'].astype(str)

    # Keep only columns needed for modeling and downstream checks
    keep_cols = ['SpecimenID', 'ToothClass', 'n_missing', 'n_present', 'n_observed', 'IsHuman', 'Age_c', 'Sex_c', 'Age', 'SexEstimate', 'Genus', 'Region']
    df = df.loc[:, [c for c in keep_cols if c in df.columns]]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial (logistic) GLM predicting antemortem tooth loss (counts) using
    IsHuman as the main predictor and controlling for age, sex, and tooth class.

    The response is provided as a two-column array of (successes, failures):
      successes = n_missing
      failures  = n_present

    We use cluster-robust standard errors clustered on SpecimenID to account for
    multiple rows per specimen (different tooth classes per specimen).

    Returns the fitted GLMResults object.
    """
    import numpy as np
    import statsmodels.api as sm

    # Make sure required columns are present
    required = ['n_missing', 'n_present', 'IsHuman', 'Age_c', 'Sex_c', 'ToothClass', 'SpecimenID']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe is missing required columns for modeling: {missing}")

    # Endog as (successes, failures) for binomial counts
    endog = np.column_stack((df['n_missing'].values, df['n_present'].values))

    # Build design matrix (exog)
    # Include intercept, IsHuman, Age_c, Sex_c, and ToothClass as categorical (drop first to avoid collinearity)
    tooth_dummies = pd.get_dummies(df['ToothClass'], prefix='Tooth', drop_first=True)

    exog = pd.concat([
        df[['IsHuman', 'Age_c', 'Sex_c']].reset_index(drop=True),
        tooth_dummies.reset_index(drop=True)
    ], axis=1)

    exog = sm.add_constant(exog, has_constant='add')

    # Fit GLM with Binomial family
    model_glm = sm.GLM(endog, exog, family=sm.families.Binomial())

    # Fit with cluster-robust standard errors (cluster by specimen to account for non-independence)
    try:
        results = model_glm.fit(cov_type='cluster', cov_kwds={'groups': df['SpecimenID']})
    except Exception:
        # Fallback to default fit if cluster covariance fails for any reason
        results = model_glm.fit()

    return results


