from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/noperturb_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Gilmore (2013) AMTL dataset into a dataframe ready for binomial regression.

    Produces the following new/guaranteed columns that the model uses:
      - num_amtl (int): number of missing teeth for the tooth-class/specimen row (keeps original name)
      - sockets (int): number of observable sockets (keeps original name)
      - prop_amtl (float): num_amtl / sockets
      - IsHuman (int): 1 if genus == 'Homo sapiens', else 0
      - age_c (float): age centered by the sample mean
      - prob_male (float): numeric probability of male (keeps original name)
      - tooth_class (category): categorical tooth class
      - specimen (category): specimen identifier (used for clustering)

    Drops rows with missing essential variables and ensures types are appropriate.
    """
    # Work on a copy
    df = df.copy()

    # Required columns for analysis
    required = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen']
    df = df.dropna(subset=required)

    # Convert numeric fields and ensure integer counts
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    # drop rows where conversion made NaN
    df = df.dropna(subset=['num_amtl', 'sockets'])

    # Ensure counts are integers and non-negative
    df['num_amtl'] = df['num_amtl'].astype(int)
    df['sockets'] = df['sockets'].astype(int)
    # Keep only rows with at least one observable socket
    df = df[df['sockets'] > 0]

    # Proportion of missing teeth in that tooth class (for interpretation and formula-based modelling)
    df['prop_amtl'] = df['num_amtl'] / df['sockets']

    # Create binary indicator for modern human vs non-human primate
    df['IsHuman'] = (df['genus'].astype(str).str.strip() == 'Homo sapiens').astype(int)

    # Center age for numerical stability
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df.dropna(subset=['age'])
    df['age_c'] = df['age'] - df['age'].mean()

    # Ensure prob_male is numeric
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')
    df = df.dropna(subset=['prob_male'])

    # Make categorical variables categorical
    df['tooth_class'] = df['tooth_class'].astype('category')
    df['specimen'] = df['specimen'].astype('category')

    # Final defensive drop (any remaining NaNs in model columns)
    model_cols = ['num_amtl', 'sockets', 'prop_amtl', 'IsHuman', 'age_c', 'prob_male', 'tooth_class', 'specimen']
    df = df.dropna(subset=model_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial regression to test whether modern humans have higher AMTL frequency than non-human primates,
    controlling for age, sex (prob_male), and tooth class. Cluster-robust standard errors are computed by specimen
    to account for non-independence of multiple tooth-class observations from the same specimen.

    Modeling choices:
      - Outcome: proportion missing (prop_amtl) modeled with Binomial family using the number of trials = sockets
        (implemented via frequency weights).
      - Primary predictor: IsHuman (1 = Homo sapiens, 0 = other genera).
      - Controls: age_c (centered age), prob_male, categorical tooth_class.
      - Clustered robust SEs by specimen.

    Returns the clustered-robust result object (statsmodels results instance with clustered covariance).
    """
    # Formula: model the proportion (num_amtl / sockets) with binomial family; include categorical tooth class
    formula = 'prop_amtl ~ IsHuman + age_c + prob_male + C(tooth_class)'

    # Fit GLM with binomial family. Use freq_weights = sockets so the model treats prop_amtl as proportions from sockets trials
    glm_model = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
    res = glm_model.fit(freq_weights=df['sockets'])

    # Obtain cluster-robust covariance estimates clustered on specimen
    # Compute clustered covariance matrix using sandwich estimator
    # sm.stats.sandwich_covariance.cov_cluster returns the clustered covariance matrix
    cluster_cov = sm.stats.sandwich_covariance.cov_cluster(res, df['specimen'])

    # Replace the covariance-related attributes on the results object so that .summary() and other methods reflect clustered SEs
    # Provide cov_params() method replacement and set bse to the clustered standard errors
    # cov_cluster returns a numpy array; ensure it's an ndarray
    cluster_cov = np.asarray(cluster_cov)
    # Override cov_params method to return the clustered covariance
    res.cov_params = lambda: cluster_cov
    # Override bse attribute to the clustered standard errors
    res.bse = np.sqrt(np.diag(cluster_cov))
    # Record cov_type and cov_kwds for reference
    res.cov_type = 'cluster'
    res.cov_kwds = {'groups': df['specimen']}

    return res