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
    Prepare data for binomial regression of AMTL.
    Produces the following columns used by the model:
      - AMTL_successes: num_amtl clipped to [0, sockets]
      - AMTL_failures: sockets - AMTL_successes
      - genus_Homo_sapiens, genus_Pongo, genus_Papio: genus indicator dummies (Pan implied reference)
      - tooth_Posterior, tooth_Premolar: tooth class dummies (Anterior implied reference)
      - age_z: standardized age
      - prob_male: sex probability (kept as provided, missing filled with 0.5)
    Rows with missing/invalid critical values (sockets <= 0, missing num_amtl/sockets/age/prob_male/genus/tooth_class) are dropped.
    """
    df = df.copy()

    # Required columns
    required = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class']
    df = df.dropna(subset=required)

    # Ensure numeric and valid sockets
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    df = df.dropna(subset=['num_amtl', 'sockets'])
    df = df[df['sockets'] > 0]

    # Clip successes to valid range [0, sockets] and compute failures
    df['AMTL_successes'] = df['num_amtl'].clip(lower=0, upper=df['sockets']).astype(int)
    df['AMTL_failures'] = (df['sockets'] - df['AMTL_successes']).astype(int)

    # Genus dummies: create explicit indicator columns. Use Pan as implicit reference (all zeros).
    df['genus'] = df['genus'].astype(str)
    df['genus_Homo_sapiens'] = (df['genus'] == 'Homo sapiens').astype(int)
    df['genus_Pongo'] = (df['genus'] == 'Pongo').astype(int)
    df['genus_Papio'] = (df['genus'] == 'Papio').astype(int)

    # Tooth-class dummies: use Anterior as reference
    df['tooth_class'] = df['tooth_class'].astype(str)
    df['tooth_Posterior'] = (df['tooth_class'] == 'Posterior').astype(int)
    df['tooth_Premolar'] = (df['tooth_class'] == 'Premolar').astype(int)

    # Standardize age (z-score). Use population std (ddof=0) for stability in small samples.
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df.dropna(subset=['age'])
    age_mean = df['age'].mean()
    age_std = df['age'].std(ddof=0)
    # Guard against zero std
    if age_std == 0 or np.isnan(age_std):
        df['age_z'] = 0.0
    else:
        df['age_z'] = (df['age'] - age_mean) / age_std

    # Ensure prob_male is numeric and fill missing with 0.5 (uninformative)
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce').fillna(0.5)

    # Final drop to ensure model columns are present and finite
    model_cols = [
        'AMTL_successes', 'AMTL_failures',
        'genus_Homo_sapiens', 'genus_Pongo', 'genus_Papio',
        'tooth_Posterior', 'tooth_Premolar',
        'age_z', 'prob_male'
    ]
    df = df.dropna(subset=model_cols)

    # Convert integer columns to ints
    df['AMTL_successes'] = df['AMTL_successes'].astype(int)
    df['AMTL_failures'] = df['AMTL_failures'].astype(int)
    for c in ['genus_Homo_sapiens', 'genus_Pongo', 'genus_Papio', 'tooth_Posterior', 'tooth_Premolar']:
        df[c] = df[c].astype(int)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a binomial (logit) GLM for AMTL counts.
    The model estimates log-odds of a tooth (in the observed socket set for a tooth class) being missing.

    Model formula (in matrix form):
      logit(p) = beta0 + beta1*genus_Homo_sapiens + beta2*genus_Pongo + beta3*genus_Papio
                 + beta4*tooth_Posterior + beta5*tooth_Premolar
                 + beta6*age_z + beta7*prob_male

    Reference categories: genus = Pan (all genus dummies = 0), tooth_class = Anterior (both tooth dummies = 0).
    Returns the fitted statsmodels GLM results object.
    """
    df = df.copy()

    # Design matrix columns used in the model
    X_cols = [
        'genus_Homo_sapiens', 'genus_Pongo', 'genus_Papio',
        'tooth_Posterior', 'tooth_Premolar',
        'age_z', 'prob_male'
    ]

    # Add constant
    X = df[X_cols]
    X = sm.add_constant(X, has_constant='add')

    # Endog as two-column array: [successes, failures]
    endog = np.column_stack((df['AMTL_successes'].values, df['AMTL_failures'].values))

    # Fit GLM (binomial family)
    model = sm.GLM(endog, X, family=sm.families.Binomial())
    results = model.fit()

    # Print a summary for immediate inspection; return results for programmatic use
    try:
        print(results.summary())
    except Exception:
        # In some headsless contexts printing the summary may fail; ignore
        pass

    return results


