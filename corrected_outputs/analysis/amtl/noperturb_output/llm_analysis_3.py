from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/amtl/noperturb_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for binomial GLM modeling.

    Output columns required for the model (and created here):
      - num_amtl: number of missing teeth in the observed tooth_class (int)
      - sockets: number of observable sockets for that tooth_class (int)
      - amtl_rate: proportion num_amtl / sockets (float)
      - genus: categorical genus (string / category), kept as-is (e.g., 'Homo sapiens')
      - tooth_class: categorical (Anterior/Posterior/Premolar)
      - age_c: centered age (age - mean(age))
      - prob_male: numeric 0-1 estimate of maleness
      - pop: categorical population/region

    The function drops rows with missing values in required fields and removes rows with sockets <= 0
    or where num_amtl > sockets (data errors). It also coerces types to appropriate dtypes.
    """

    # Make a copy to avoid mutating input
    df = df.copy()

    # Required columns for analysis
    required_cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'pop']
    # Drop rows missing any required columns
    df = df.dropna(subset=required_cols)

    # Ensure sockets is positive integer; drop rows with non-positive sockets
    df = df[df['sockets'] > 0]

    # Ensure num_amtl is non-negative integer and not greater than sockets
    # If any num_amtl > sockets, drop those rows as data errors
    df = df[df['num_amtl'] >= 0]
    df = df[df['num_amtl'] <= df['sockets']]

    # Create proportion column for modeling (endog for binomial model)
    df['amtl_rate'] = df['num_amtl'] / df['sockets']

    # Convert categorical columns to category dtype (helps with modeling and ensures consistent behavior)
    df['genus'] = df['genus'].astype('category')
    df['tooth_class'] = df['tooth_class'].astype('category')
    df['pop'] = df['pop'].astype('category')

    # Center age to improve interpretability and model fitting
    df['age_c'] = df['age'] - df['age'].mean()

    # Ensure prob_male is numeric and bounded 0-1; clip small numeric noise
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')
    df = df.dropna(subset=['prob_male'])
    df['prob_male'] = df['prob_male'].clip(0.0, 1.0)

    # Final sanity checks: keep only rows with finite rates
    df = df[np.isfinite(df['amtl_rate'])]

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a binomial (logit) GLM for AMTL using counts of missing teeth out of sockets.

    Model specification (primary):
      amtl_rate ~ C(genus, Treatment(reference='Homo sapiens')) + age_c + prob_male + C(tooth_class) + C(pop)

    We use the proportion amtl_rate as the endogenous variable and pass sockets as weights
    so that the model is equivalent to modeling num_amtl ~ Binomial(sockets, p).

    Returns a dictionary with the fitted GLM results and a robust-covariance version of the results.
    """

    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    import numpy as np

    # Ensure required columns are present
    needed = ['amtl_rate', 'sockets', 'genus', 'age_c', 'prob_male', 'tooth_class', 'pop']
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found in dataframe")

    # Formula: set Homo sapiens as the reference category for genus
    # Use categorical tooth_class and pop as fixed effects
    formula = 'amtl_rate ~ C(genus, Treatment(reference="Homo sapiens")) + age_c + prob_male + C(tooth_class) + C(pop)'

    # Fit GLM with Binomial family and use sockets as weights (number of trials)
    # Using the proportion as endog and weights=number of trials yields the binomial likelihood
    glm_binom = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), weights=df['sockets'])
    results = glm_binom.fit()

    # Compute robust (sandwich) covariance results for inference robust to some model misspecification
    try:
        results_robust = results.get_robustcov_results(cov_type='HC3')
    except Exception:
        # Fallback to original results if robust covariance calculation fails
        results_robust = results

    # Calculate dispersion (for binomial, dispersion ~ 1 under correct specification).
    # For GLM-binomial with weights, an empirical dispersion can be estimated as
    # sum(resid_pearson**2) / residual_df
    pearson_resid = results.resid_pearson
    rdf = float(results.df_resid)
    dispersion = np.nan
    if rdf > 0:
        dispersion = np.sum(pearson_resid**2) / rdf

    return {
        'glm_results': results,
        'glm_results_robust': results_robust,
        'dispersion': dispersion,
        'formula': formula
    }


