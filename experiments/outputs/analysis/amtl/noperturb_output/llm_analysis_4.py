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
    Prepare the dataset for binomial regression / GEE analysis.
    - Drops rows with missing critical fields
    - Removes rows with non-positive sockets
    - Computes proportion of missing teeth (prop_amtl)
    - Creates binary is_human indicator (Homo sapiens vs others)
    - Centers age (age_c)
    - Ensures tooth_class is categorical
    Returns a dataframe containing columns used by the model:
    ['num_amtl', 'sockets', 'prop_amtl', 'is_human', 'age', 'age_c', 'prob_male', 'tooth_class', 'specimen', 'genus', 'pop']
    """
    df = df.copy()

    # Required columns for analysis
    required_cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Drop rows with missing values in critical columns
    df = df.dropna(subset=required_cols)

    # Remove rows where sockets is not positive
    df = df[df['sockets'] > 0]

    # Ensure integer counts
    # (some datasets may store them as floats; cast carefully)
    df['num_amtl'] = df['num_amtl'].astype(int)
    df['sockets'] = df['sockets'].astype(int)

    # Proportion of teeth missing in the observed sockets
    df['prop_amtl'] = df['num_amtl'] / df['sockets']

    # Binary indicator for modern humans (Homo sapiens)
    # Use exact match to the expected label 'Homo sapiens' but be robust to whitespace
    df['genus'] = df['genus'].astype(str).str.strip()
    df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

    # Center age for model interpretability
    df['age_c'] = df['age'] - df['age'].mean()

    # Ensure tooth_class is categorical and clean whitespace
    df['tooth_class'] = df['tooth_class'].astype(str).str.strip()
    df['tooth_class'] = pd.Categorical(df['tooth_class'], categories=pd.unique(df['tooth_class']))

    # Keep commonly useful columns and return
    keep_cols = ['num_amtl', 'sockets', 'prop_amtl', 'is_human', 'genus', 'age', 'age_c', 'prob_male', 'tooth_class', 'specimen', 'pop']
    # Some columns (e.g., pop) may be missing in some inputs; keep those that exist
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a clustered binomial model comparing AMTL frequency in modern humans versus non-human primates,
    controlling for age (centered), sex probability, and tooth class.

    Approach:
    - Use Generalized Estimating Equations (GEE) with binomial family to model counts/proportions
      while accounting for correlation among observations from the same specimen.
    - Endog is the proportion prop_amtl and sockets are provided as weights (binomial denominator).
    - Exchangeable working correlation is used for within-specimen correlation.

    Returns the fitted GEE results object.
    """
    # Basic checks
    for c in ['num_amtl', 'sockets', 'prop_amtl', 'is_human', 'age_c', 'prob_male', 'tooth_class', 'specimen']:
        if c not in df.columns:
            raise ValueError(f"Transformed dataframe must contain column: {c}")

    # Construct formula: main test is is_human (1 = Homo sapiens)
    formula = 'prop_amtl ~ is_human + age_c + prob_male + C(tooth_class)'

    # Instantiate GEE with binomial family. Use sockets as weights (the binomial denominator).
    # Group by specimen to account for multiple tooth-class observations per individual.
    model_gee = sm.GEE.from_formula(
        formula,
        groups='specimen',
        data=df,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
        weights=df['sockets']
    )

    result = model_gee.fit()

    # Print a brief summary for quick inspection; return full result for downstream use
    print(result.summary())
    return result


