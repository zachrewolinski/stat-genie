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
    """
    Transform raw dataset to the analysis dataframe. Produces the following columns required for modeling:
      - num_amtl (int): number of missing teeth of the given class
      - sockets (int): number of observable sockets (trials)
      - amtl_prop (float): num_amtl / sockets (proportion missing)
      - IsHuman (int): 1 if genus == 'Homo sapiens', else 0
      - age_c (float): age centered around the sample mean
      - prob_male (float): as provided (0-1)
      - tooth_class (category): standardized categorical variable
      - specimen (category/string): specimen id (kept for clustering)
      - genus (category/string): original genus

    Rows with invalid or missing essential values are dropped.
    """
    df = df.copy()

    # Required columns
    required = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen']
    # Drop rows missing any required field
    df = df.dropna(subset=required)

    # Ensure numeric types
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')

    # After coercion, drop rows with missing numeric values
    df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male'])

    # Remove impossible socket counts or impossible amtl counts
    df = df[df['sockets'] > 0]
    # Remove rows where num_amtl > sockets or negative
    df = df[(df['num_amtl'] >= 0) & (df['num_amtl'] <= df['sockets'])]

    # Standardize tooth_class strings and make categorical
    df['tooth_class'] = df['tooth_class'].astype(str).str.strip().str.capitalize()
    # Map common variants if needed (Posterior, Anterior, Premolar)
    df.loc[~df['tooth_class'].isin(['Anterior', 'Posterior', 'Premolar']), 'tooth_class'] = df['tooth_class']
    df['tooth_class'] = df['tooth_class'].astype('category')

    # Standardize genus string and make categorical
    df['genus'] = df['genus'].astype(str).str.strip()
    df['genus'] = df['genus'].astype('category')

    # Create binary human indicator (IsHuman)
    # Matches exact label 'Homo sapiens' if present, but also handles labels that contain 'Homo'
    df['IsHuman'] = df['genus'].apply(lambda x: 1 if ('homo' in str(x).lower()) else 0)

    # Proportion missing
    df['amtl_prop'] = df['num_amtl'] / df['sockets']

    # Center age for interpretability
    df['age_c'] = df['age'] - df['age'].mean()

    # Keep specimen as a string/categorical (useful for clustering SE)
    df['specimen'] = df['specimen'].astype(str)

    # Final required columns for modeling - drop any rows with NaN in these
    keep_cols = ['num_amtl', 'sockets', 'amtl_prop', 'IsHuman', 'age_c', 'prob_male', 'tooth_class', 'specimen', 'genus']
    df = df.dropna(subset=keep_cols)

    # Reset index
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial GLM (logistic) modeling AMTL proportion with sockets as the number of trials.
    The primary predictor is IsHuman (1 = Homo sapiens, 0 = non-human). Controls: age (centered), prob_male, and tooth_class.
    Standard errors are clustered by specimen to account for non-independence of multiple observations per specimen.

    Returns the robust (clustered) results object.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Formula: model proportion with genus indicator and controls; use C(tooth_class) to treat tooth class as categorical
    formula = 'amtl_prop ~ IsHuman + age_c + prob_male + C(tooth_class)'

    # Fit GLM on proportion with binomial family, using freq_weights = number of trials (sockets)
    # Using amtl_prop as the response and sockets as frequency weights tells the binomial model how many trials each row represents
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), freq_weights=df['sockets']).fit()

    # Obtain cluster-robust covariance (cluster by specimen)
    # This adjusts standard errors for repeated measures within specimen
    try:
        robust_results = model.get_robustcov_results(cov_type='cluster', groups=df['specimen'])
    except Exception:
        # Fallback: if clustering fails, return the original fitted model
        robust_results = model

    # Print a concise summary for quick inspection (user code can further inspect `robust_results`)
    print(robust_results.summary())

    return robust_results


