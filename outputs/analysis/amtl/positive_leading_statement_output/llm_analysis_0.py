from typing import Any
import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.sandwich_covariance as sw
import matplotlib.pyplot as plt
import pickle

# Example read (left as in original; users may replace with their path)
# df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/positive_leading_statement_output/amtl.csv')


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for binomial modeling of AMTL.

    Steps performed:
    - Copy the input to avoid side-effects.
    - Drop rows missing essential fields (num_amtl, sockets, age, prob_male, genus, tooth_class, specimen).
    - Remove observations with sockets <= 0 (no trials to observe AMTL).
    - Ensure num_amtl <= sockets by capping if necessary.
    - Compute prop_amtl = num_amtl / sockets.
    - Create binary indicator is_human: 1 if genus == 'Homo sapiens', else 0.
    - Convert tooth_class to a categorical column tooth_class_cat.
    - Create a standardized age variable age_scaled (mean 0, sd 1) for model stability.
    - Create weights column equal to sockets (number of trials) for the binomial GLM.
    - Return a dataframe containing only the columns used in modeling.
    """
    df = df.copy()

    # Drop rows missing essential variables
    essential = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen']
    df = df.dropna(subset=essential)

    # Remove observations with zero or negative sockets
    df = df[df['sockets'] > 0].copy()

    # Ensure numeric counts and that missing count does not exceed sockets
    # Keep original as numeric; cast safely
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    # Cap num_amtl at sockets and ensure non-negative
    df['num_amtl'] = df[['num_amtl', 'sockets']].min(axis=1).fillna(0)
    df['num_amtl'] = df['num_amtl'].clip(lower=0).astype(int)
    df['sockets'] = df['sockets'].astype(float)

    # Proportion missing in the observed sockets
    df['prop_amtl'] = df['num_amtl'] / df['sockets']

    # Primary independent variable: is the specimen a modern human?
    # Use exact string 'Homo sapiens' per dataset schema
    df['is_human'] = (df['genus'].astype(str).str.strip() == 'Homo sapiens').astype(int)

    # Categorical tooth class as used in model formula
    df['tooth_class_cat'] = df['tooth_class'].astype('category')

    # Standardize age (mean 0, sd 1) for numerical stability and interpretability
    age_mean = df['age'].mean()
    age_std = df['age'].std(ddof=0)
    if pd.isna(age_mean):
        df['age_scaled'] = pd.NA
    else:
        if age_std == 0 or pd.isna(age_std):
            df['age_scaled'] = df['age'] - age_mean
        else:
            df['age_scaled'] = (df['age'] - age_mean) / age_std

    # Keep stdev_age (uncertainty in age) as a control; create if missing
    if 'stdev_age' not in df.columns:
        df['stdev_age'] = pd.NA

    # Weights for binomial GLM: number of trials (sockets)
    df['weights'] = df['sockets'].astype(float)

    # Final column selection (only columns used by the model)
    cols = [
        'specimen',
        'num_amtl',
        'sockets',
        'prop_amtl',
        'is_human',
        'age',
        'age_scaled',
        'prob_male',
        'stdev_age',
        'tooth_class_cat',
        'genus',
        'weights'
    ]

    # Ensure all required columns exist in the final dataframe
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

    return df[cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial regression to test whether modern humans have higher AMTL frequency than non-human primates,
    adjusting for age, sex (prob_male), and tooth class. Cluster-robust standard errors by specimen
    are used to account for multiple observations per specimen.

    Modeling approach:
    - Use a GLM (binomial family) on the observed proportion prop_amtl with weights = sockets (number of trials).
    - Include is_human as the primary predictor (1 = Homo sapiens, 0 = non-human primate).
    - Control for age (age_scaled), prob_male, and categorical tooth_class_cat.
    - Obtain cluster-robust standard errors by specimen to account for within-specimen correlation.

    Returns the fitted results object with cluster robust SEs applied where possible.
    """
    # Ensure required columns exist
    required = ['prop_amtl', 'weights', 'specimen', 'is_human', 'age_scaled', 'prob_male', 'tooth_class_cat']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Formula: proportion of missing teeth explained by is_human and controls
    formula = 'prop_amtl ~ is_human + age_scaled + prob_male + C(tooth_class_cat)'

    # Fit GLM with Binomial family. Use weights = sockets (number of trials) so the model treats prop_amtl appropriately.
    glm_model = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), weights=df['weights'])

    # Try to have the fit method directly produce clustered covariances (preferred)
    try:
        res = glm_model.fit(cov_type='cluster', cov_kwds={'groups': df['specimen']})
        res_cluster = res
    except TypeError:
        # Some statsmodels versions may not accept cov_type in fit for GLM.
        # Fit normally and then compute clustered sandwich covariance manually and attach it.
        res = glm_model.fit()
        try:
            cluster_cov = sw.cov_cluster(res, df['specimen'])
            # Attach the clustered covariance as the default covariance used by summary/cov_params
            res.cov_params_default = cluster_cov
            res.cov_type = 'cluster'
            res.cov_kwds = {'groups': df['specimen']}
            res_cluster = res
        except Exception:
            # If cluster covariance computation fails, fall back to the original result without clustered SEs
            res_cluster = res

    # Compute odds ratio and 95% CI for the is_human coefficient using the covariance in res_cluster if available
    params = res_cluster.params
    # Try to obtain covariance matrix for computing SEs; prefer cov_params_default if present
    try:
        cov = getattr(res_cluster, 'cov_params_default', None)
        if cov is None:
            cov = res_cluster.cov_params()
        se_series = pd.Series(np.sqrt(np.diag(cov)), index=params.index)
    except Exception:
        # Last resort: use internal bse if available
        if hasattr(res_cluster, 'bse'):
            se_series = res_cluster.bse
        else:
            se_series = pd.Series(np.nan, index=params.index)

    if 'is_human' in params.index and ('is_human' in se_series.index):
        coef = params['is_human']
        se = se_series['is_human']
        or_point = np.exp(coef)
        if np.isfinite(se):
            ci_lower = np.exp(coef - 1.96 * se)
            ci_upper = np.exp(coef + 1.96 * se)
            # Compute p-value using normal approximation
            z = coef / se if se != 0 else np.nan
            pval = 2 * (1 - scipy.stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
            print(f"is_human coef: {coef:.4f}, OR={or_point:.3f}, 95% CI=[{ci_lower:.3f}, {ci_upper:.3f}], p={pval:.4g}")
        else:
            print(f"is_human coef: {coef:.4f}, OR={or_point:.3f}, SE unavailable to compute CI/p-value")
    else:
        print("Warning: 'is_human' not in fitted model parameters or SEs unavailable")

    # Print full clustered-summary if possible; otherwise print the regular summary
    try:
        print(res_cluster.summary())
    except Exception:
        try:
            print(res.summary())
        except Exception:
            print("Could not produce model summary output.")

    # Return the result object (with clustered covariance attached if computed)
    return res_cluster