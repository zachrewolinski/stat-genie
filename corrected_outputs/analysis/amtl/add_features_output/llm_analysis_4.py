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
    Transform the raw dataset for binomial modeling of antemortem tooth loss (AMTL).

    Produces the following columns required by the model:
      - prop_amtl: proportion of missing teeth within the observed sockets (num_amtl / sockets)
      - sockets: number of observable sockets (trials for binomial model)
      - num_amtl: number of missing teeth (successes)
      - genus: cleaned genus factor
      - tooth_class: cleaned tooth class factor
      - age_scaled: standardized age (mean 0, sd 1)
      - prob_male: probability specimen is male (kept as-is, 0-1)
      - specimen: specimen identifier (for clustering)

    Rows with insufficient data (e.g., missing sockets or zero sockets) are removed.
    """
    # Work on a copy
    df = df.copy()

    # Keep necessary columns and drop rows missing required fields
    required = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen']
    df = df.dropna(subset=required)

    # Ensure sockets are integer and positive
    # Some datasets might have floats for sockets; coerce to int when it's near-integer
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    df = df.dropna(subset=['sockets'])
    # Remove rows with zero or negative sockets (can't model binomial trials = 0)
    df = df[df['sockets'] >= 1]
    # Ensure num_amtl numeric and bounded between 0 and sockets
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce').fillna(0).astype(int)
    # Clip improbable values
    df['num_amtl'] = df['num_amtl'].clip(lower=0, upper=df['sockets'])

    # Create proportion outcome (successes / trials)
    df['prop_amtl'] = df['num_amtl'] / df['sockets']

    # Clean genus and tooth_class strings and convert to categorical
    df['genus'] = df['genus'].astype(str).str.strip()
    df['tooth_class'] = df['tooth_class'].astype(str).str.strip()

    # Standardize age for modeling (mean 0, sd 1). Keep numeric coercion.
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    # Drop rows where age became missing after coercion
    df = df.dropna(subset=['age'])
    age_mean = df['age'].mean()
    age_std = df['age'].std(ddof=0) if df['age'].std(ddof=0) != 0 else 1.0
    df['age_scaled'] = (df['age'] - age_mean) / age_std

    # Ensure prob_male is numeric and clipped to [0,1]
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')
    df['prob_male'] = df['prob_male'].clip(lower=0.0, upper=1.0)
    df = df.dropna(subset=['prob_male'])

    # Ensure specimen id is string (for clustering later)
    df['specimen'] = df['specimen'].astype(str)

    # Final housekeeping: cast tooth_class and genus to category dtype
    df['tooth_class'] = df['tooth_class'].astype('category')
    df['genus'] = df['genus'].astype('category')

    # Return only columns needed for modeling + a few helpful originals
    keep_cols = [
        'specimen', 'genus', 'tooth_class', 'num_amtl', 'sockets', 'prop_amtl',
        'age', 'age_scaled', 'prob_male'
    ]
    # Some columns may not exist if input missing; filter
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit a binomial (logistic) model for AMTL proportion with clustering by specimen.

    The model estimates how genus (particularly Homo sapiens) relates to the probability
    of a tooth being missing in its class, controlling for age, sex (prob_male), and tooth class.

    Modeling approach:
      - Use the proportion prop_amtl as the response and provide the number of trials via var_weights
        in statsmodels' GLM to represent grouped binomial data (successes/trials).
      - Include genus and tooth_class as categorical predictors with explicit treatment contrasts.
      - Obtain cluster-robust standard errors clustered on specimen to account for multiple rows per specimen.

    Returns the fitted results object with cluster-robust covariances applied.
    """
    import patsy
    import statsmodels.api as sm

    # Ensure required columns are present
    required = ['prop_amtl', 'sockets', 'genus', 'age_scaled', 'prob_male', 'tooth_class', 'specimen']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Define formula: compare genus levels (reference = 'Pan' (chimpanzees))
    # and use Posterior tooth class as reference for tooth_class
    formula = (
        "prop_amtl ~ C(genus, Treatment(reference='Pan')) + age_scaled + prob_male + "
        "C(tooth_class, Treatment(reference='Posterior'))"
    )

    # Build design matrices using patsy (keeps categorical encoding consistent)
    y, X = patsy.dmatrices(formula, data=df, return_type='dataframe')

    # y is the proportion; pass number of trials as var_weights to GLM
    model_glm = sm.GLM(y, X, family=sm.families.Binomial(), var_weights=df['sockets'].values)
    res = model_glm.fit()

    # Obtain cluster-robust standard errors by specimen to account for within-specimen correlation
    # (multiple tooth classes recorded per specimen)
    try:
        res_cluster = res.get_robustcov_results(cov_type='cluster', groups=df['specimen'])
    except Exception:
        # Fallback: if clustering fails, return the original fit but warn the user
        import warnings
        warnings.warn('Clustered SE calculation failed; returning non-clustered GLM results.')
        res_cluster = res

    # Print concise summary and return the robust-results object
    print(res_cluster.summary())
    return res_cluster


