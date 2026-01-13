from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/add_features_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Required columns for analysis
    required = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen']
    # Drop rows with missing values in required fields
    df = df.dropna(subset=required)

    # Keep only rows with at least one observable socket (trials must be > 0)
    df = df[df['sockets'] > 0]

    # Standardize/clean and create modeling columns
    # Counts and trials
    df['NumMissing'] = df['num_amtl'].astype(int)
    df['Sockets'] = df['sockets'].astype(int)

    # Proportion of teeth missing (for GLM with binomial family using trial weights)
    df['prop'] = df['NumMissing'] / df['Sockets']

    # Binary indicator for Homo sapiens (1 if genus contains 'Homo' or exact 'Homo sapiens')
    # Use case-insensitive matching to be robust to formatting
    df['Genus_Homo'] = df['genus'].astype(str).str.contains('Homo', case=False, na=False).astype(int)

    # Tooth class as a clean categorical column
    df['ToothClass'] = df['tooth_class'].astype(str)

    # Standardize continuous controls (z-scores). Use ddof=0 for population-style standardization.
    df['Age_z'] = (df['age'] - df['age'].mean()) / (df['age'].std(ddof=0) if df['age'].std(ddof=0) != 0 else 1.0)
    df['ProbMale_z'] = (df['prob_male'] - df['prob_male'].mean()) / (df['prob_male'].std(ddof=0) if df['prob_male'].std(ddof=0) != 0 else 1.0)

    # Ensure specimen is a string for clustering
    df['specimen'] = df['specimen'].astype(str)

    # Return only columns needed for modeling (plus originals if desired)
    keep_cols = ['NumMissing', 'Sockets', 'prop', 'Genus_Homo', 'Age_z', 'ProbMale_z', 'ToothClass', 'specimen']
    # Also preserve original columns in case downstream checks are needed
    for c in df.columns:
        if c not in keep_cols:
            pass
    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Work on a copy
    df = df.copy()

    # Ensure categorical coding of ToothClass
    df['ToothClass'] = df['ToothClass'].astype('category')

    # Formula: model the observed proportion with binomial family and use 'Sockets' as frequency weights
    # This is equivalent to modeling NumMissing ~ covariates with Binomial trials = Sockets
    formula = 'prop ~ Genus_Homo + Age_z + ProbMale_z + C(ToothClass)'

    # Fit GLM for binomial data using proportions with freq_weights = number of trials
    # Note: using freq_weights (Socket counts) tells the model how many Bernoulli trials each observation represents
    glm_mod = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
    glm_res = glm_mod.fit(freq_weights=df['Sockets'])

    # Obtain cluster-robust standard errors clustered by specimen to account for non-independence
    try:
        clustered_res = glm_res.get_robustcov_results(cov_type='cluster', groups=df['specimen'])
    except Exception:
        # If clustering fails for any reason, fall back to the original fit
        clustered_res = glm_res

    # Return the clustered results object (has .summary())
    return clustered_res


