from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster
from scipy import stats


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the dataframe used for modeling.

    Produces the following columns required by the model:
      - MajorityChoice: binary (1 if y==2 (majority), else 0)
      - age_c: standardized age (mean 0, sd 1)
      - culture: categorical site identifier (kept as category dtype)
      - is_female: 1 if gender == 1 (girl), 0 if gender == 2 (boy)
      - majority_first: ensured to be numeric 0/1
      - religiousness: kept as-is (numeric ordinal)
      - school: kept for clustering

    Drops rows with missing values in any of the required columns.
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Required raw columns: y, age, culture, gender, majority_first, religiousness, school
    required_cols = ['y', 'age', 'culture', 'gender', 'majority_first', 'religiousness', 'school']

    # Drop rows missing required variables
    df = df.dropna(subset=required_cols)

    # Dependent variable: 1 if chose the majority option (y == 2), 0 otherwise
    df['MajorityChoice'] = (df['y'] == 2).astype(int)

    # Standardize age (mean 0, sd 1) to improve numeric stability and interpretability
    # If age has zero variance (unlikely here), fall back to centered age
    if df['age'].std(ddof=0) > 0:
        df['age_c'] = (df['age'] - df['age'].mean()) / df['age'].std(ddof=0)
    else:
        df['age_c'] = df['age'] - df['age'].mean()

    # Culture as categorical factor
    df['culture'] = df['culture'].astype('category')

    # Gender: encode 1 = female (girl), 0 = male (boy). Original coding: 1=girl, 2=boy
    # If other codes present, treat non-1 as 0 after informing through NA handling above
    df['is_female'] = (df['gender'] == 1).astype(int)

    # Ensure majority_first is numeric 0/1
    # Convert to numeric allowing NA; we'll drop NA rows later and then cast to int safely
    df['majority_first'] = pd.to_numeric(df['majority_first'], errors='coerce')

    # Ensure religiousness is numeric
    df['religiousness'] = pd.to_numeric(df['religiousness'], errors='coerce')

    # After transformations, drop any rows that became NA (e.g., non-numeric majority_first or religiousness)
    model_cols = ['MajorityChoice', 'age_c', 'culture', 'is_female', 'majority_first', 'religiousness', 'school']
    df = df.dropna(subset=model_cols)

    # Now that NA rows are removed, cast majority_first to integer (0/1)
    # It's safe because rows with NA were dropped above
    df['majority_first'] = df['majority_first'].astype(int)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression predicting the probability of choosing the majority option.

    Model specification (fixed effects):
      MajorityChoice ~ age_c * C(culture) + is_female + majority_first + religiousness

    Interaction age_c * C(culture) tests whether the developmental slope differs across cultural sites.
    We fit a logistic regression (Logit) and obtain school-clustered robust standard errors to account
    for within-school dependence.

    Returns the model results object with clustered robust covariance when possible.
    """
    # Ensure required columns exist
    required = ['MajorityChoice', 'age_c', 'culture', 'is_female', 'majority_first', 'religiousness', 'school']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe is missing required columns for modeling: {missing}")

    # Formula: main effect of age_c, culture (categorical), their interaction, and controls
    formula = 'MajorityChoice ~ age_c * C(culture) + is_female + majority_first + religiousness'

    # Fit Logit (binary outcome)
    logit_fit = smf.logit(formula=formula, data=df).fit(disp=False)

    # Obtain cluster-robust covariance by school (if there are at least 2 clusters)
    n_clusters = df['school'].nunique()
    if n_clusters >= 2:
        try:
            # Compute clustered covariance matrix
            clustered_cov = cov_cluster(logit_fit, df['school'])
            # Attach cluster-robust results to the fitted object for downstream use
            # bse, tvalues, and pvalues computed using normal approximation
            bse_cluster = np.sqrt(np.diag(clustered_cov))
            tvalues_cluster = logit_fit.params / bse_cluster
            pvalues_cluster = 2 * stats.norm.sf(np.abs(tvalues_cluster))

            # Attach attributes to the results object
            logit_fit.cov_cluster = clustered_cov
            logit_fit.bse_cluster = bse_cluster
            logit_fit.tvalues_cluster = tvalues_cluster
            logit_fit.pvalues_cluster = pvalues_cluster

            results = logit_fit
        except Exception:
            # If anything fails when computing clustered covariance, return the plain fit
            results = logit_fit
    else:
        results = logit_fit  # cannot cluster with <2 clusters

    # Return the fitted results (with robust covariances accessible if clustering applied)
    return results