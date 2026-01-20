from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/amtl/add_features_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for binomial modeling of AMTL.

    Produces the following important columns used in the model:
      - AMTL_successes: integer count of missing teeth for the given tooth class (num_amtl)
      - sockets: integer count of observable sockets for the given tooth class (sockets)
      - AMTL_failures: sockets - AMTL_successes (for diagnostics)
      - AMTL_prop: AMTL_successes / sockets (proportion missing)
      - IsHuman: 1 if genus indicates Homo sapiens, 0 otherwise
      - age: numeric age at death (converted)
      - ProbMale: numeric probability of male sex (converted)
      - tooth_class: standardized categorical tooth class

    Rows with missing essential fields or invalid sockets are removed.
    """
    df = df.copy()

    # Keep only rows with the essential fields
    required = ['num_amtl', 'sockets', 'genus', 'age', 'prob_male', 'tooth_class']
    df = df.dropna(subset=required)

    # Convert numeric fields to numeric types
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['ProbMale'] = pd.to_numeric(df['prob_male'], errors='coerce')

    # Drop rows that became NaN from coercion
    df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'ProbMale'])

    # Sockets must be a positive integer and num_amtl must be between 0 and sockets
    # Force integer rounding for counts (but only after validation)
    df['sockets'] = df['sockets'].astype(int)
    # Negative or zero sockets cannot be modeled; remove
    df = df[df['sockets'] > 0]

    # Round num_amtl to integer and clamp to valid range
    df['num_amtl'] = df['num_amtl'].round().astype(int)
    df = df[(df['num_amtl'] >= 0) & (df['num_amtl'] <= df['sockets'])]

    # Create AMTL columns used for modeling
    df['AMTL_successes'] = df['num_amtl'].astype(int)
    df['AMTL_failures'] = (df['sockets'] - df['AMTL_successes']).astype(int)
    df['AMTL_prop'] = df['AMTL_successes'] / df['sockets']

    # Binary indicator for modern human specimens
    # Use case-insensitive matching; handle possible variants like 'Homo sapiens', 'Homo', etc.
    df['genus_str'] = df['genus'].astype(str).str.strip().str.lower()
    df['IsHuman'] = df['genus_str'].apply(lambda x: 1 if ('homo sapiens' in x or x == 'homo' or x.startswith('homo')) else 0)
    df = df.drop(columns=['genus_str'])

    # Standardize tooth_class categories to a consistent set (Title case expected values)
    df['tooth_class'] = df['tooth_class'].astype(str).str.strip().str.title()

    # Keep only well-known tooth_class values (if other values exist, they will remain but C() in formula will handle them)
    # Final columns to return: include specimen identifier if present for diagnostics
    keep_cols = [c for c in ['specimen', 'tooth_class', 'AMTL_successes', 'AMTL_failures', 'sockets', 'AMTL_prop', 'IsHuman', 'age', 'ProbMale'] if c in df.columns]

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial (logit link) GLM to model probability of AMTL while controlling for age, sex, and tooth class.

    Model formula (proportion with var_weights = sockets):
      AMTL_prop ~ IsHuman + age + ProbMale + C(tooth_class)

    We use the proportion (AMTL_prop) as the dependent variable and pass the number of sockets as variance weights
    so that the model treats each row as a binomial observation with different denominators.

    Returns the fitted GLMResults object from statsmodels.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    df = df.copy()

    # Ensure required columns are present
    required = ['AMTL_prop', 'sockets', 'IsHuman', 'age', 'ProbMale', 'tooth_class']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Fit GLM using proportion outcome with var_weights = sockets to model a binomial process
    formula = 'AMTL_prop ~ IsHuman + age + ProbMale + C(tooth_class)'

    # Use the formula API; supply var_weights to indicate binomial denominators
    glm_model = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), var_weights=df['sockets'])
    results = glm_model.fit()

    # It's often useful to see robust (clustered) SEs if specimens are not independent across rows,
    # but that requires a cluster variable (e.g., specimen) and is optional. For now return the standard results.
    return results


