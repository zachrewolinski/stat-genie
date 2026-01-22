from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster, cov_hc3
from scipy import stats

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/add_features_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw AMTL dataset into analysis-ready dataframe.

    Outputs (columns used in modeling):
      - AMTL_successes: integer, number of missing teeth for the row
      - AMTL_trials: integer, number of observable sockets for the row (trials)
      - AMTL_rate: float, AMTL_successes / AMTL_trials (proportion)
      - IsHuman: binary indicator (1 if genus == 'Homo sapiens', else 0)
      - age: numeric (copied, ensures numeric dtype)
      - prob_male: numeric (copied, ensures numeric dtype)
      - tooth_class: categorical with preserved levels
      - specimen: identifier (kept for possible clustering)
    """
    df = df.copy()

    # Keep only columns we need (but preserve others if present)
    required_cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Ensure numeric types for key columns
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')

    # Drop rows with missing critical information
    df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen'])

    # Ensure integer counts and valid trials
    # Remove rows where sockets <= 0 (no observable trials)
    df = df[df['sockets'] > 0].copy()

    # Round counts to integers (if float) and clip num_amtl to [0, sockets]
    df['AMTL_trials'] = df['sockets'].round().astype(int)
    df['AMTL_successes'] = df['num_amtl'].round().astype(int)
    df['AMTL_successes'] = df['AMTL_successes'].clip(lower=0, upper=df['AMTL_trials'])

    # Proportion for binomial modeling; will use trials as weights
    # Avoid division by zero (AMTL_trials already filtered >0)
    df['AMTL_rate'] = df['AMTL_successes'] / df['AMTL_trials']

    # Create binary human indicator (explicit string match)
    # Accept common spelling in dataset: 'Homo sapiens'
    df['IsHuman'] = (df['genus'].astype(str).str.strip() == 'Homo sapiens').astype(int)

    # Normalize/clean tooth_class values and set as categorical
    df['tooth_class'] = df['tooth_class'].astype(str).str.strip()
    # Optionally standardize capitalization
    df['tooth_class'] = df['tooth_class'].str.capitalize()
    # Ensure expected categories appear (if there are variants, they will still be treated as levels)
    df['tooth_class'] = pd.Categorical(df['tooth_class'], categories=['Anterior', 'Premolar', 'Posterior'])

    # Keep only columns needed for modeling (plus specimen for clustering)
    keep_cols = ['AMTL_successes', 'AMTL_trials', 'AMTL_rate', 'IsHuman', 'age', 'prob_male', 'tooth_class', 'specimen']
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial (logistic) GLM to test whether modern humans have higher AMTL rates than
    non-human primates, controlling for age, sex (prob_male), and tooth class.

    Modeling approach:
      - Outcome: AMTL_rate (proportion) with Binomial family and weights=AMTL_trials
      - Predictor of interest: IsHuman (1 = Homo sapiens, 0 = non-human)
      - Controls: age (continuous), prob_male (continuous), tooth_class (categorical)
      - Clustered (robust) standard errors at the specimen level to account for repeated measures
        or non-independence within specimens.

    Returns:
      - results_robust: an object exposing params, bse, cov_params, tvalues/zvalues, pvalues and a summary() method.
    """
    # Basic formula: proportion outcome with categorical tooth_class
    formula = 'AMTL_rate ~ IsHuman + age + prob_male + C(tooth_class)'

    # Fit GLM with Binomial family; use AMTL_trials as weights so the model treats observations
    # as proportions with differing denominators.
    model_glm = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), weights=df['AMTL_trials'])
    results = model_glm.fit()

    # Compute cluster-robust standard errors clustered by specimen if possible,
    # otherwise fall back to HC3 robust SEs.
    try:
        cov = cov_cluster(results, df['specimen'])
    except Exception:
        cov = cov_hc3(results)

    # Build a lightweight wrapper around the original results that exposes
    # the robust covariance, standard errors, z/t-values, p-values, and a summary method.
    class RobustResultsWrapper:
        def __init__(self, orig_res, cov_matrix):
            self._orig = orig_res
            self.cov_params = cov_matrix
            # Ensure cov is a numpy array
            self.cov_params = np.asarray(self.cov_params)
            # Align dimensions in case of mismatch
            self.params = orig_res.params
            # Compute standard errors from robust covariance
            self.bse = np.sqrt(np.diag(self.cov_params))
            # Protect against zeros
            self.bse = np.where(self.bse == 0, np.nan, self.bse)
            # z (or t) values and two-sided p-values using normal approx
            self.zvalues = self.params / self.bse
            self.pvalues = 2 * (1 - stats.norm.cdf(np.abs(self.zvalues)))

        def summary(self):
            # Create a concise table similar to statsmodels' coef table
            try:
                coef_table = pd.DataFrame({
                    'coef': self.params,
                    'std err': self.bse,
                    'z': self.zvalues,
                    'P>|z|': self.pvalues
                })
                # include original model's index order
                coef_table.index = self.params.index
                header = f"Model: {self._orig.model.__class__.__name__}\n"
                return header + coef_table.to_string(float_format=lambda x: f"{x:0.4f}")
            except Exception:
                # Fallback to original results summary if anything unexpected happens
                return str(self._orig.summary())

        # Expose repr to allow printing directly
        def __repr__(self):
            return self.summary()

    results_robust = RobustResultsWrapper(results, cov)

    # Print a short summary and return the robust results object
    print(results_robust.summary())
    return results_robust