from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/anonymize_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for binomial GLM modeling of AMTL.

    Input raw columns expected:
      - feature1: tooth class (Anterior/Posterior/Premolar)
      - feature2: specimen id
      - feature3: number of teeth missing of given class
      - feature4: number of observable sockets (trials)
      - feature5: estimated age at death
      - feature6: age uncertainty
      - feature7: sex estimate/probability
      - feature8: genus (Homo sapiens, Pan, Pongo, Papio)
      - feature9: region

    Returns dataframe with the following columns required by the model:
      - MissingCount (int)  -- successes
      - TotalSockets (int)  -- trials
      - PresentCount (int)  -- TotalSockets - MissingCount
      - Age (float)          -- original age
      - Age_z (float)        -- standardized age (mean 0, sd 1)
      - AgeUncertainty (float)
      - SexProb (float)      -- sex estimate (kept continuous)
      - Genus (category)
      - ToothClass (category)
      - SpecimenID (object)
      - Region (object)
      - IsHuman (int)        -- indicator for Homo sapiens (1) vs others (0)
    """
    df = df.copy()

    # Rename raw columns to meaningful names
    rename_map = {
        'feature1': 'ToothClass',
        'feature2': 'SpecimenID',
        'feature3': 'MissingCount',
        'feature4': 'TotalSockets',
        'feature5': 'Age',
        'feature6': 'AgeUncertainty',
        'feature7': 'SexProb',
        'feature8': 'Genus',
        'feature9': 'Region'
    }
    df = df.rename(columns=rename_map)

    # Drop rows missing essential fields for AMTL calculation or key covariates
    df = df.dropna(subset=['MissingCount', 'TotalSockets', 'Age', 'Genus', 'ToothClass'])

    # Ensure numeric types for counts and coerce if necessary
    df['MissingCount'] = pd.to_numeric(df['MissingCount'], errors='coerce').fillna(0).astype(int)
    df['TotalSockets'] = pd.to_numeric(df['TotalSockets'], errors='coerce').fillna(0).astype(int)

    # Correct inconsistent rows where MissingCount > TotalSockets by capping
    mask_bad = df['MissingCount'] > df['TotalSockets']
    if mask_bad.any():
        df.loc[mask_bad, 'MissingCount'] = df.loc[mask_bad, 'TotalSockets']

    # Create PresentCount (failures) for binomial modeling
    df['PresentCount'] = df['TotalSockets'] - df['MissingCount']

    # Create binary indicator for Homo sapiens
    df['Genus'] = df['Genus'].astype(str).str.strip()
    df['IsHuman'] = (df['Genus'] == 'Homo sapiens').astype(int)

    # Standardize age (center and scale). Use population sd (ddof=0) for stability.
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    age_mean = df['Age'].mean()
    age_std = df['Age'].std(ddof=0) if df['Age'].std(ddof=0) > 0 else 1.0
    df['Age_z'] = (df['Age'] - age_mean) / age_std

    # Coerce SexProb and AgeUncertainty to numeric and fill missing with medians
    df['SexProb'] = pd.to_numeric(df['SexProb'], errors='coerce')
    if df['SexProb'].isna().any():
        df['SexProb'] = df['SexProb'].fillna(df['SexProb'].median())

    df['AgeUncertainty'] = pd.to_numeric(df['AgeUncertainty'], errors='coerce')
    if df['AgeUncertainty'].isna().any():
        df['AgeUncertainty'] = df['AgeUncertainty'].fillna(df['AgeUncertainty'].median())

    # Coerce categories to categorical dtype and try to set a meaningful reference order for Genus
    # We'll prefer Pan as reference when present (Pan is often a common comparative baseline).
    df['Genus'] = df['Genus'].astype('category')
    try:
        desired_order = [g for g in ['Pan', 'Pongo', 'Papio', 'Homo sapiens'] if g in df['Genus'].cat.categories]
        if len(desired_order) > 0:
            df['Genus'] = pd.Categorical(df['Genus'], categories=desired_order, ordered=False)
    except Exception:
        pass

    df['ToothClass'] = df['ToothClass'].astype('category')

    # Keep only necessary columns for modeling (model function can compute dummies)
    out_cols = ['MissingCount', 'TotalSockets', 'PresentCount', 'Age', 'Age_z', 'AgeUncertainty',
                'SexProb', 'Genus', 'ToothClass', 'SpecimenID', 'Region', 'IsHuman']
    # Ensure all requested columns exist (some may be missing if input incomplete)
    for c in out_cols:
        if c not in df.columns:
            df[c] = pd.NA

    return df[out_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a binomial (logistic) GLM for AMTL (MissingCount out of TotalSockets) as a function of Genus,
    controlling for Age, Sex, ToothClass. Cluster-robust standard errors are calculated at the specimen level
    (SpecimenID) to account for multiple tooth-class observations per specimen.

    Returns:
      - results: statsmodels GLMResultsWrapper with cluster-robust covariances applied when possible.
    """
    import statsmodels.api as sm
    import pandas as pd

    # Drop rows with zero trials (no observable sockets) because they provide no binomial information
    df_model = df.copy()
    df_model = df_model[df_model['TotalSockets'] > 0]

    if df_model.shape[0] == 0:
        raise ValueError('No observations with TotalSockets > 0 available for modeling.')

    # Prepare endog as 2-column array [successes, failures]
    endog = df_model[['MissingCount', 'PresentCount']].to_numpy()

    # Build design matrix (exog)
    # Use categorical dummies for Genus and ToothClass (drop_first=True to avoid multicollinearity)
    genus_dummies = pd.get_dummies(df_model['Genus'].astype(str), prefix='Genus', drop_first=True)
    tooth_dummies = pd.get_dummies(df_model['ToothClass'].astype(str), prefix='Tooth', drop_first=True)

    # Numeric covariates: standardized age, sex probability, age uncertainty, and IsHuman indicator
    numeric_covs = df_model[['Age_z', 'SexProb', 'AgeUncertainty', 'IsHuman']].copy()

    # Concatenate all predictors
    exog = pd.concat([numeric_covs.reset_index(drop=True), genus_dummies.reset_index(drop=True),
                      tooth_dummies.reset_index(drop=True)], axis=1)

    # Add constant (intercept)
    exog = sm.add_constant(exog, has_constant='add')

    # Fit binomial GLM using a 2-column endog for successes/failures
    glm_binom = sm.GLM(endog, exog, family=sm.families.Binomial())
    res = glm_binom.fit()

    # Attempt to obtain cluster-robust standard errors clustered by SpecimenID
    # If SpecimenID has a single unique value for all rows, clustering will fail; guard for that case.
    try:
        groups = df_model['SpecimenID']
        if groups.nunique() > 1:
            res_cluster = res.get_robustcov_results(cov_type='cluster', groups=groups)
        else:
            # Not enough clusters to cluster; return the original fitted result
            res_cluster = res
    except Exception:
        # If anything goes wrong obtaining clustered covariances, return the original fitted result
        res_cluster = res

    # Print brief model summary for user inspection
    try:
        print(res_cluster.summary())
    except Exception:
        # summary printing is optional; continue to return results
        pass

    return res_cluster


