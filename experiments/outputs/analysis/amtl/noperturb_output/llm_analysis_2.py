from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices


df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/noperturb_output/amtl.csv')


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataset for binomial GLM modeling of AMTL.
    Transformations performed:
    - Drop rows missing critical columns
    - Remove rows with nonpositive sockets or impossible counts (num_amtl > sockets)
    - Standardize age to age_z (z-score)
    - Create proportion column prop_amtl for diagnostics (num_amtl / sockets)
    - Clean categorical columns (strip whitespace)

    Final dataframe contains at minimum the columns used in the model:
    ['num_amtl', 'sockets', 'genus', 'age_z', 'prob_male', 'tooth_class', 'pop', 'specimen', 'prop_amtl']
    """

    df = df.copy()

    # Required columns
    required = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'pop', 'specimen']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Drop rows with missing values in required columns (raw presence)
    df = df.dropna(subset=required)

    # Ensure numeric types where appropriate
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')

    # Drop rows where sockets are missing or nonpositive
    df = df.dropna(subset=['sockets'])
    df = df[df['sockets'] > 0]

    # Remove impossible counts where num_amtl is missing, negative or greater than sockets
    df = df.dropna(subset=['num_amtl'])
    df = df[(df['num_amtl'] >= 0) & (df['num_amtl'] <= df['sockets'])]

    # Ensure prob_male is within [0,1]; drop otherwise
    df = df.dropna(subset=['prob_male'])
    df = df[(df['prob_male'] >= 0) & (df['prob_male'] <= 1)]

    # Standardize (z-score) age for numerical stability
    # Use population std (ddof=0)
    age_mean = df['age'].mean()
    age_std = df['age'].std(ddof=0)
    if age_std == 0 or np.isnan(age_std):
        df['age_z'] = 0.0
    else:
        df['age_z'] = (df['age'] - age_mean) / age_std

    # Proportion for diagnostics and potential plotting
    df['prop_amtl'] = df['num_amtl'] / df['sockets']

    # Clean categorical text fields
    df['genus'] = df['genus'].astype(str).str.strip()
    df['tooth_class'] = df['tooth_class'].astype(str).str.strip()
    df['pop'] = df['pop'].astype(str).str.strip()
    df['specimen'] = df['specimen'].astype(str).str.strip()

    # Optionally restrict to the four genera of interest if other taxa exist
    allowed_genera = ['Homo sapiens', 'Pan', 'Pongo', 'Papio']
    df = df[df['genus'].isin(allowed_genera)].copy()

    # Reindex and return
    df = df.reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial GLM for AMTL (num_amtl out of sockets) as a function of genus
    while controlling for age (standardized), sex probability (prob_male), and tooth class.

    Model specification (formula):
      prop_amtl ~ C(genus, Treatment(reference="Homo sapiens")) + age_z + prob_male + C(tooth_class)

    The model is fit with a Binomial family using socket counts as frequency weights so that
    the response is interpreted as successes/trials. After fitting, cluster-robust standard errors
    (by population 'pop') are computed and returned when possible.

    The function returns the fitted results object with cluster-robust cov if available.
    """

    # Ensure required columns exist
    needed = ['num_amtl', 'sockets', 'genus', 'age_z', 'prob_male', 'tooth_class', 'pop', 'prop_amtl']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Transformed dataframe is missing required columns: {missing}")

    # Build formula using the proportion column created in transform.
    # Use patsy to construct design matrices so we can supply a clipped response to avoid exact 0/1 issues.
    formula = 'prop_amtl ~ C(genus, Treatment(reference="Homo sapiens")) + age_z + prob_male + C(tooth_class)'

    # Prepare design matrices
    y, X = dmatrices(formula, data=df, return_type='dataframe')

    # y is a DataFrame with one column; convert to 1d array
    y_vals = np.asarray(y.iloc[:, 0]).astype(float)

    # Clip responses away from exact 0 or 1 to avoid boundary issues in the initial deviance calculation.
    eps = 1e-6
    y_clipped = np.clip(y_vals, eps, 1 - eps)

    # Use GLM with Binomial family and frequency weights equal to number of sockets (trials)
    # freq_weights ensures each proportion is treated as aggregated over sockets trials.
    model_glm = sm.GLM(y_clipped, X, family=sm.families.Binomial(), freq_weights=df['sockets'].astype(float))

    try:
        res = model_glm.fit()
    except Exception:
        # As a fallback, try using statsmodels' formula API with the clipped proportions inplace.
        # Create a temporary column for the clipped response (internal helper only)
        df_temp = df.copy()
        df_temp['_prop_clipped_for_model'] = y_clipped
        try:
            res = smf.glm(formula='_prop_clipped_for_model ~ C(genus, Treatment(reference="Homo sapiens")) + age_z + prob_male + C(tooth_class)',
                          data=df_temp,
                          family=sm.families.Binomial(),
                          weights=df_temp['sockets']).fit()
        except Exception as e:
            # Re-raise a clearer error
            raise RuntimeError("GLM fitting failed even after clipping the response.") from e

    # Attempt to get cluster-robust covariance by population to account for within-population correlation
    try:
        res_clust = res.get_robustcov_results(cov_type='cluster', groups=df['pop'])
    except Exception:
        res_clust = res

    # Print a brief summary for user convenience; the returned object contains full results
    print(res_clust.summary())

    return res_clust