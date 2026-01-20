from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/amtl/replace_with_rvs_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataset for binomial GLM of AMTL.

    Produces the following new/clean columns used in modeling:
      - prop_amtl: proportion missing = num_amtl / sockets
      - IsHuman: 1 if genus is 'Homo sapiens', else 0
      - Age_z: standardized age
      - ProbMale: copy of prob_male for clarity
      - ToothClass: categorical tooth class (consistent categories)

    Drops rows with missing required fields and rows with non-positive socket counts.
    """
    df = df.copy()

    # Drop rows missing essential variables
    df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class'])

    # Ensure we only keep rows where there is at least one observable socket
    df = df[df['sockets'] > 0]

    # Compute proportion of AMTL for convenience (used as endog in GLM with weights)
    df['prop_amtl'] = df['num_amtl'] / df['sockets']

    # Create binary indicator for modern humans (Homo sapiens)
    # Match robustly by lowercasing and trimming whitespace
    df['IsHuman'] = (df['genus'].astype(str).str.strip().str.lower() == 'homo sapiens').astype(int)

    # Standardize age (z-score) to aid interpretation and numeric stability
    # Use population-like std (ddof=0) for centering; ddof default in pandas is 1 so specify if desired
    df['Age_z'] = (df['age'] - df['age'].mean()) / df['age'].std(ddof=0)

    # Copy probability-of-male column to a clear name used in modeling
    df['ProbMale'] = df['prob_male']

    # Ensure tooth class is categorical and set a consistent ordering (keeps original levels if missing will remain)
    df['ToothClass'] = df['tooth_class'].astype('category')
    try:
        # prefer this ordering but if dataset lacks one of the levels, set_categories will still work
        df['ToothClass'] = df['ToothClass'].cat.set_categories(['Anterior', 'Premolar', 'Posterior'])
    except Exception:
        # if set_categories fails for any reason, keep whatever categories exist
        df['ToothClass'] = df['ToothClass'].astype('category')

    # Keep only columns required for modeling plus identifiers for traceability
    # (retain specimen for potential clustering or diagnostics)
    keep_cols = ['specimen', 'num_amtl', 'sockets', 'prop_amtl', 'IsHuman', 'Age_z', 'ProbMale', 'ToothClass', 'genus', 'pop']
    df = df.loc[:, [c for c in keep_cols if c in df.columns]]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial GLM predicting proportion of AMTL (num_amtl / sockets) using binomial family.

    Model formula:
      prop_amtl ~ IsHuman + Age_z + ProbMale + C(ToothClass)

    The model uses 'sockets' as weights (number of trials) while the outcome is the proportion prop_amtl.
    Returns the fitted GLMResults object.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Ensure required columns exist
    required = ['prop_amtl', 'sockets', 'IsHuman', 'Age_z', 'ProbMale', 'ToothClass']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Define formula: proportion outcome with tooth class as categorical predictor
    formula = 'prop_amtl ~ IsHuman + Age_z + ProbMale + C(ToothClass)'

    # Fit GLM with Binomial family. Use 'sockets' as weights to specify number of trials.
    # Using the proportion as the response and sockets as weights implements a binomial model for successes/trials.
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), weights=df['sockets'])
    results = model.fit()

    # Print a concise summary for immediate inspection; return the results object for programmatic access
    print(results.summary())

    return results


