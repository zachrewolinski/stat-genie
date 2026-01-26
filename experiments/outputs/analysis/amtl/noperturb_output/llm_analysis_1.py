from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/noperturb_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Clean missing data for variables required in the binomial/GEE model.
    - Create a binary indicator 'Human' for genus == 'Homo sapiens'.
    - Create proportion outcome 'prop_amtl' = num_amtl / sockets.
    - Standardize continuous controls (age, prob_male) to z-scores: age_z, prob_male_z.
    - Ensure tooth_class and specimen are categorical where appropriate.

    Returns a dataframe that contains all columns referenced by the model function.
    """
    df = df.copy()

    # Required columns
    required_cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Drop rows missing any of the key variables (no way to include them in model)
    df = df.dropna(subset=required_cols)

    # Keep only rows with positive (non-zero) sockets to avoid division/mis-specification
    df = df[df['sockets'] > 0].copy()

    # Create Human indicator (1 = Homo sapiens, 0 = non-human primate)
    # Normalize genus strings to avoid mismatch due to whitespace
    df['genus'] = df['genus'].astype(str).str.strip()
    df['Human'] = (df['genus'] == 'Homo sapiens').astype(int)

    # Proportion of missing teeth in the scored tooth class
    df['prop_amtl'] = df['num_amtl'] / df['sockets']

    # Standardize continuous controls (z-scores). Use population std (ddof=0) for modeling stability
    df['age_z'] = (df['age'] - df['age'].mean()) / (df['age'].std(ddof=0) if df['age'].std(ddof=0) != 0 else 1.0)
    df['prob_male_z'] = (df['prob_male'] - df['prob_male'].mean()) / (df['prob_male'].std(ddof=0) if df['prob_male'].std(ddof=0) != 0 else 1.0)

    # Ensure tooth_class and specimen are categorical types
    df['tooth_class'] = df['tooth_class'].astype('category')
    df['specimen'] = df['specimen'].astype('category')

    # Optional: drop any extreme proportions outside [0,1] due to bad data
    df = df[(df['prop_amtl'] >= 0.0) & (df['prop_amtl'] <= 1.0)]

    # Final dataframe contains the columns the modeling code expects
    # (num_amtl, sockets, prop_amtl, Human, age_z, prob_male_z, tooth_class, specimen)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a binomial regression for AMTL using GEE to account for multiple tooth-class observations per specimen.

    Model specification (primary):
      prop_amtl ~ Human + age_z + prob_male_z + C(tooth_class)

    - Uses GEE with Binomial family and logit link.
    - Uses 'sockets' as the binomial denominator via var_weights.
    - Groups by 'specimen' with an exchangeable working correlation structure to account for within-specimen dependence.

    Returns the fitted GEE results object.
    """
    import statsmodels.api as sm

    # Check required columns
    required = ['prop_amtl', 'sockets', 'Human', 'age_z', 'prob_male_z', 'tooth_class', 'specimen']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Transformed dataframe is missing required columns for modeling: {missing}")

    # Formula: proportion modeled with binomial family; tooth_class as categorical
    formula = 'prop_amtl ~ Human + age_z + prob_male_z + C(tooth_class)'

    # Build and fit GEE. Use var_weights = sockets (number of trials per observation)
    cov_struct = sm.cov_struct.Exchangeable()
    gee_model = sm.GEE.from_formula(
        formula,
        groups='specimen',
        data=df,
        family=sm.families.Binomial(),
        cov_struct=cov_struct,
        var_weights=df['sockets']
    )

    result = gee_model.fit()

    # Print a concise summary; return the result object for further inspection
    print(result.summary())
    return result


