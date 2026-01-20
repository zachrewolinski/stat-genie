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
    Transform raw dataset to a cleaned dataframe with the columns required for modeling.

    Input columns (raw): feature1..feature9 per provided schema
    Mapping performed:
      - feature1 -> ToothClass
      - feature2 -> SpecimenID
      - feature3 -> MissingCount (number of teeth missing of given class)
      - feature4 -> N_Sockets (number of observable sockets / trials)
      - feature5 -> Age (estimated age at death)
      - feature6 -> AgeUncertainty (uncertainty of age estimate)
      - feature7 -> SexMaleProb (estimate / probability of maleness)
      - feature8 -> Genus (Pan, Pongo, Homo sapiens, Papio, ...)
      - feature9 -> Region

    Derived columns returned (and used in model):
      - PresentCount = N_Sockets - MissingCount
      - IsHomo = 1 if Genus contains 'Homo' (case-insensitive), else 0
      - Age_c = Age - mean(Age) (centered age)

    Cleaning steps:
      - Drop rows with missing essential fields
      - Drop rows with N_Sockets <= 0
      - Drop rows where MissingCount > N_Sockets (biologically/recording inconsistent)
      - Standardize ToothClass categories and make categorical
    """
    df = df.copy()

    # Rename raw columns to clear names
    df = df.rename(columns={
        'feature1': 'ToothClass',
        'feature2': 'SpecimenID',
        'feature3': 'MissingCount',
        'feature4': 'N_Sockets',
        'feature5': 'Age',
        'feature6': 'AgeUncertainty',
        'feature7': 'SexMaleProb',
        'feature8': 'Genus',
        'feature9': 'Region'
    })

    # Drop rows missing core variables required for analysis
    df = df.dropna(subset=['MissingCount', 'N_Sockets', 'ToothClass', 'Genus', 'Age', 'SexMaleProb'])

    # Ensure numeric types for counts and age
    df['MissingCount'] = pd.to_numeric(df['MissingCount'], errors='coerce')
    df['N_Sockets'] = pd.to_numeric(df['N_Sockets'], errors='coerce')
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['SexMaleProb'] = pd.to_numeric(df['SexMaleProb'], errors='coerce')
    df['AgeUncertainty'] = pd.to_numeric(df['AgeUncertainty'], errors='coerce')

    # Drop rows that became NA after coercion
    df = df.dropna(subset=['MissingCount', 'N_Sockets', 'Age', 'SexMaleProb'])

    # Remove rows with zero or negative number of sockets (cannot model binomial)
    df = df[df['N_Sockets'] > 0]

    # Remove biologically/recording inconsistent rows where MissingCount > N_Sockets
    df = df[df['MissingCount'] <= df['N_Sockets']]

    # Compute present count (non-missing teeth observed)
    df['PresentCount'] = df['N_Sockets'] - df['MissingCount']

    # Create binary indicator for Homo (modern humans). We use a case-insensitive substring match to catch 'Homo sapiens' or similar labels.
    df['IsHomo'] = df['Genus'].astype(str).str.contains('Homo', case=False, na=False).astype(int)

    # Center age for interpretability
    df['Age_c'] = df['Age'] - df['Age'].mean()

    # Standardize ToothClass labels and make categorical. Ensure expected categories are present.
    df['ToothClass'] = df['ToothClass'].astype(str).str.strip().str.title()

    # Map common variants if present
    df['ToothClass'] = df['ToothClass'].replace({
        'Premolar': 'Premolar',
        'Posterior': 'Posterior',
        'Anterior': 'Anterior'
    })

    # Force categorical type (order is arbitrary; drop-first encoding will be used in modeling)
    df['ToothClass'] = pd.Categorical(df['ToothClass'], categories=['Anterior', 'Premolar', 'Posterior'])

    # Keep and return only the columns needed for modeling and interpretation
    out_cols = ['SpecimenID', 'ToothClass', 'MissingCount', 'PresentCount', 'N_Sockets', 'Age', 'AgeUncertainty', 'Age_c', 'SexMaleProb', 'IsHomo', 'Genus', 'Region']
    return df[out_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial (logistic) GLM predicting the probability of a tooth being missing (AMTL)
    as a function of whether the specimen is Homo (IsHomo), age (centered), sex-probability
    (SexMaleProb), and tooth class (ToothClass). The model uses the counts format for a
    binomial GLM: endog = [MissingCount, PresentCount].

    Returns:
      - results: fitted statsmodels GLMResults object (so the caller can examine coefficients, CIs, tests)
    """
    # Work on a copy
    df = df.copy()

    # Build tooth-class dummy variables (drop first to avoid multicollinearity)
    tooth_dummies = pd.get_dummies(df['ToothClass'], prefix='Tooth', drop_first=True)

    # Exogenous variables: intercept + IsHomo + Age_c + SexMaleProb + tooth dummies
    exog = pd.concat([df[['IsHomo', 'Age_c', 'SexMaleProb']].reset_index(drop=True), tooth_dummies.reset_index(drop=True)], axis=1)
    exog = sm.add_constant(exog, has_constant='add')

    # Endogenous for binomial counts: 2-column array of successes and failures
    endog = np.vstack([df['MissingCount'].values, df['PresentCount'].values]).T

    # Fit GLM with binomial family (logit link by default)
    glm_binom = sm.GLM(endog, exog, family=sm.families.Binomial())
    results = glm_binom.fit()

    # Print a concise summary to stdout and return the full results object for downstream inspection
    print(results.summary())
    return results


