from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm


df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/noperturb_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to the form required for binomial GLM modeling of AMTL.

    Produces the following new columns used in the model:
      - prop_amtl: num_amtl / sockets (proportion of missing teeth for the observation)
      - age_c: centered age (age - mean(age))

    Also ensures categorical columns are strings and removes rows with invalid/missing key data.
    """
    # Work on a copy
    df = df.copy()

    # Required columns for modeling
    required_cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen']
    # Drop rows missing any of the required columns
    df = df.dropna(subset=required_cols)

    # Ensure numeric columns are numeric
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')

    # Drop newly introduced NaNs (bad conversions)
    df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male'])

    # Keep only rows where sockets > 0 (valid binomial trials)
    df = df[df['sockets'] > 0]

    # Ensure num_amtl in valid range [0, sockets]
    df = df[(df['num_amtl'] >= 0) & (df['num_amtl'] <= df['sockets'])]

    # Ensure prob_male within [0, 1]; drop otherwise
    df = df[(df['prob_male'] >= 0.0) & (df['prob_male'] <= 1.0)]

    # Convert categorical columns to strings (explicit) to avoid issues in formula handling
    df['genus'] = df['genus'].astype(str)
    df['tooth_class'] = df['tooth_class'].astype(str)
    df['specimen'] = df['specimen'].astype(str)
    if 'pop' in df.columns:
        df['pop'] = df['pop'].astype(str)

    # Derived response: proportion of missing teeth per observation
    df['prop_amtl'] = df['num_amtl'] / df['sockets']

    # Center age to make the intercept interpretable
    df['age_c'] = df['age'] - df['age'].mean()

    # Final safety drop: drop any rows with NaN created during transformations
    df = df.dropna(subset=['prop_amtl', 'age_c'])

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial (logistic) GLM for AMTL rates and produce clustered robust standard errors by specimen.

    Model specification (formula):
      prop_amtl ~ C(genus, Treatment(reference="Homo sapiens")) + age_c + prob_male + C(tooth_class)

    The binomial response is supplied as a proportion (prop_amtl) with frequency weights equal to the number of sockets
    so that the GLM fits the number of successes out of the number of trials.

    Returns a dictionary with the raw fitted GLM result and a clustered-robust-covariance adjusted result
    (clustered by 'specimen'). The raw GLM result is a statsmodels result object; the clustered result provides
    parameter estimates and cluster-robust standard errors and related statistics.
    """
    # Check required columns
    for col in ['prop_amtl', 'sockets', 'genus', 'age_c', 'prob_male', 'tooth_class', 'specimen']:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe passed to model().")

    # Formula: use Treatment coding with 'Homo sapiens' as the reference level
    formula = 'prop_amtl ~ C(genus, Treatment(reference="Homo sapiens")) + age_c + prob_male + C(tooth_class)'

    # Fit GLM with Binomial family. Use freq_weights = number of trials (sockets)
    glm_model = smf.glm(formula=formula,
                        data=df,
                        family=sm.families.Binomial(),
                        freq_weights=df['sockets'])

    res = glm_model.fit()

    # Attempt to compute cluster-robust covariance by specimen.
    clustered_res = None

    # First try the built-in convenience method if available
    try:
        clustered_res = res.get_robustcov_results(cov_type='cluster', groups=df['specimen'])
    except Exception:
        # Fallback: compute cluster-robust covariance matrix using sandwich estimator
        try:
            from statsmodels.stats.sandwich_covariance import cov_cluster
            cov = cov_cluster(res, df['specimen'])
        except Exception:
            # If cov_cluster is not available, raise informative error
            raise RuntimeError("Could not compute cluster-robust covariance: neither "
                               "res.get_robustcov_results nor statsmodels.stats.sandwich_covariance.cov_cluster "
                               "are available in this environment.")

        # Build a lightweight wrapper object that exposes common attributes/methods expected by users.
        class ClusteredResults:
            def __init__(self, base_res, cov_matrix):
                self._base = base_res
                # params as pandas Series (keep index)
                self.params = base_res.params.copy()
                # covariance as DataFrame for nicer labeling if params is a Series
                try:
                    self.cov_params = pd.DataFrame(cov_matrix, index=self.params.index, columns=self.params.index)
                except Exception:
                    # Fallback to numpy array if indices don't align
                    self.cov_params = cov_matrix
                # standard errors
                self.bse = pd.Series(np.sqrt(np.diag(cov_matrix)), index=self.params.index)
                # z-statistics and p-values using normal approximation
                with np.errstate(divide='ignore', invalid='ignore'):
                    z = self.params.values / self.bse.values
                z = np.asarray(z)
                # handle zeros in bse leading to inf z; pvals become 0 in that case
                pvals = 2 * (1 - norm.cdf(np.abs(z)))
                self.pvalues = pd.Series(pvals, index=self.params.index)
                self.df_model = getattr(base_res, 'df_model', None)
                self.df_resid = getattr(base_res, 'df_resid', None)

            def cov_params_default(self):
                # for compatibility if some code expects method
                return self.cov_params

            def conf_int(self, alpha=0.05):
                q = norm.ppf(1 - alpha / 2.0)
                lower = self.params - q * self.bse
                upper = self.params + q * self.bse
                ci = pd.DataFrame({'lower': lower, 'upper': upper})
                return ci

            def summary(self):
                # Return the base model summary; this will show original SEs, but users can inspect
                # params, bse, pvalues, and conf_int on this wrapper for clustered results.
                try:
                    return self._base.summary()
                except Exception:
                    return f"ClusteredResults for model with params:\n{self.params}"

            # Provide a fallback repr
            def __repr__(self):
                return (f"<ClusteredResults: params={list(self.params.index)}, "
                        f"n_params={len(self.params)}>")

        clustered_res = ClusteredResults(res, cov)

    return {
        'glm_result': res,
        'clustered_result': clustered_res
    }