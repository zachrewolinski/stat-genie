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
    Transform raw dataset into analysis-ready dataframe.

    Inputs (original columns expected):
      - feature1: tooth class (Anterior/Posterior/Premolar)
      - feature2: specimen id
      - feature3: number of teeth missing of given class
      - feature4: number of observable sockets for that class (trials)
      - feature5: estimated age at death
      - feature6: age uncertainty
      - feature7: sex estimate (numeric between 0 and 1)
      - feature8: genus (Homo sapiens, Pan, Pongo, Papio)
      - feature9: region

    Outputs (columns used in modeling):
      - n_missing, n_observed, prop_missing, IsHuman, Genus,
        ToothClass, Age, Age_c, Age_c2, SexEstimate, SexF, Region,
        SpecimenID, AgeUncertainty
    """
    df = df.copy()

    # Rename raw features to meaningful names
    df['ToothClass'] = df['feature1'].astype(str)
    df['SpecimenID'] = df['feature2']

    # Counts of missing teeth and number observed (trials). Coerce to numeric and integer if possible.
    df['n_missing'] = pd.to_numeric(df['feature3'], errors='coerce')
    df['n_observed'] = pd.to_numeric(df['feature4'], errors='coerce')

    # Age and uncertainty
    df['Age'] = pd.to_numeric(df['feature5'], errors='coerce')
    df['AgeUncertainty'] = pd.to_numeric(df['feature6'], errors='coerce')

    # Sex estimate: numeric score (e.g., probability or estimate). Keep original and derive binary.
    df['SexEstimate'] = pd.to_numeric(df['feature7'], errors='coerce')

    # Genus and region
    df['Genus'] = df['feature8'].astype(str)
    df['Region'] = df['feature9'].astype(str)

    # Basic cleaning: drop rows without observed sockets or missing counts
    df = df[~df['n_observed'].isna() & ~df['n_missing'].isna()]

    # Ensure integer counts where sensible; floor/round missing counts, ensure non-negative
    df['n_observed'] = df['n_observed'].round().astype(int)
    df['n_missing'] = df['n_missing'].round().astype(int)

    # Remove impossible rows
    df = df[df['n_observed'] > 0]
    df.loc[df['n_missing'] < 0, 'n_missing'] = 0
    # If n_missing > n_observed (recording error), cap to n_observed
    df.loc[df['n_missing'] > df['n_observed'], 'n_missing'] = df.loc[df['n_missing'] > df['n_observed'], 'n_observed']

    # Proportion missing (dependent variable in binomial model). Keep for modeling convenience.
    df['prop_missing'] = df['n_missing'] / df['n_observed']

    # Primary IV: human vs non-human. Use exact match for 'Homo sapiens' but allow small variations.
    df['IsHuman'] = df['Genus'].apply(lambda x: 1 if isinstance(x, str) and ('Homo' in x or 'Homo sapiens' in x) else 0)

    # Derive binary sex indicator: SexF = 1 for female-like estimates (SexEstimate >= 0.5), 0 for male-like (< 0.5), NaN preserved if missing
    df['SexF'] = df['SexEstimate'].apply(lambda v: (1 if (not pd.isna(v) and v >= 0.5) else (0 if (not pd.isna(v) and v < 0.5) else np.nan)))

    # Center age and add quadratic term to model nonlinearity
    df['Age_c'] = df['Age'] - df['Age'].median()
    df['Age_c2'] = df['Age_c'] ** 2

    # Keep only columns needed for analysis (but return full transformed df for downstream checks)
    keep_cols = ['SpecimenID', 'Genus', 'IsHuman', 'ToothClass', 'n_missing', 'n_observed', 'prop_missing',
                 'Age', 'AgeUncertainty', 'Age_c', 'Age_c2', 'SexEstimate', 'SexF', 'Region']
    df = df.loc[:, [c for c in keep_cols if c in df.columns]]

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a binomial (logistic) generalized linear model for AMTL proportion.

    Model specification:
      prop_missing ~ IsHuman + C(ToothClass) + Age_c + Age_c2 + SexF
    Binomial family with weights equal to number of observable sockets (n_observed).

    Returns the fitted GLMResults object (statsmodels).
    """
    import statsmodels.formula.api as smf

    # Work on a copy
    df = df.copy()

    # Drop rows where required model inputs are missing
    # We require n_missing, n_observed, ToothClass, and Age (Age can be missing in some specimens; drop those for primary model)
    model_df = df.dropna(subset=['n_missing', 'n_observed', 'ToothClass', 'Age_c'])

    # If SexF is missing for some rows, allow them (statsmodels will handle NaN by dropping rows); alternatively we could impute.
    # Build formula. Use C(ToothClass) to treat tooth class as categorical.
    formula = 'prop_missing ~ IsHuman + C(ToothClass) + Age_c + Age_c2 + SexF'

    # Fit binomial GLM using proportion as endog and weights = n_observed for binomial trials
    # Note: statsmodels will interpret the dependent variable as a proportion and the 'weights' argument as number of trials
    fit = smf.glm(formula=formula,
                  data=model_df,
                  family=sm.families.Binomial(),
                  weights=model_df['n_observed']).fit()

    # Return the fitted results object. The caller can inspect fit.summary() or fit.params etc.
    return fit


