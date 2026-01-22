from typing import Any
import numpy as np
import pandas as pd
import sklearn  # noqa: F401
import scipy  # noqa: F401
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt  # noqa: F401
import pickle  # noqa: F401


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for binomial regression.

    Produces the following columns required by the model:
      - num_amtl (int): number of missing teeth for the tooth class
      - sockets (int): number of observable sockets (trials)
      - proportion (float): num_amtl / sockets (response for GLM with weights=sockets)
      - IsHuman (int): 1 if genus indicates Homo sapiens, 0 otherwise
      - age_c (float): age centered around the sample mean
      - prob_male (float): clipped to [0,1]
      - tooth_class (category): categorical tooth class
      - pop (category): population/provenance
      - specimen (category/string): specimen id (kept for clustering)
    """
    df = df.copy()

    # Required source columns
    required = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen', 'pop']
    # Drop rows missing essential data
    df = df.dropna(subset=required)

    # Ensure numeric types where appropriate
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')

    # Drop rows that became NA after coercion
    df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male'])

    # Keep only rows with positive number of sockets
    df = df[df['sockets'] > 0].copy()

    # Ensure counts are integers and valid (0 <= num_amtl <= sockets)
    df['num_amtl'] = df['num_amtl'].round().astype(int)
    # Remove any rows where num_amtl > sockets or num_amtl < 0 as data errors
    df = df[(df['num_amtl'] >= 0) & (df['num_amtl'] <= df['sockets'])].copy()

    # Proportion outcome for binomial GLM (we will pass sockets as weights)
    df['proportion'] = df['num_amtl'] / df['sockets']

    # Create primary independent variable: human vs non-human primate
    # Accept values like 'Homo sapiens', 'Homo', etc. Case-insensitive matching to 'homo'
    df['IsHuman'] = df['genus'].astype(str).str.lower().str.contains('homo').astype(int)

    # Center age (helps interpretation and numeric stability)
    df['age_c'] = df['age'] - df['age'].mean()

    # Clip prob_male to [0,1] in case of rounding or input error
    df['prob_male'] = df['prob_male'].clip(0.0, 1.0)

    # Ensure categorical columns are of category dtype
    df['tooth_class'] = df['tooth_class'].astype('category')
    df['pop'] = df['pop'].astype('category')
    # specimen kept as identifier (string/category) for clustering standard errors
    df['specimen'] = df['specimen'].astype(str)

    # Final safety: drop any remaining rows with missing model columns
    model_cols = ['num_amtl', 'sockets', 'proportion', 'IsHuman', 'age_c', 'prob_male', 'tooth_class', 'pop', 'specimen']
    df = df.dropna(subset=model_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)
    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial (logistic-link) GLM to test whether Homo sapiens have higher AMTL rates
    than non-human primates, controlling for age, sex (prob_male), tooth class, and population.

    Uses the proportion (num_amtl / sockets) as the response and supplies sockets as weights
    so the GLM models binomial counts. Returns an object exposing parameter estimates and
    cluster-robust standard errors clustered by specimen (to account for multiple tooth-class
    rows per specimen).
    """
    import statsmodels.formula.api as smf_local
    from scipy import stats as _stats

    # Formula: proportion modeled as a function of IsHuman (primary IV) + controls.
    # C(tooth_class) and C(pop) denote categorical fixed effects.
    formula = 'proportion ~ IsHuman + age_c + prob_male + C(tooth_class) + C(pop)'

    # Fit the GLM using Binomial family and pass sockets as weights (number of trials)
    glm_binom = smf_local.glm(formula=formula,
                              data=df,
                              family=sm.families.Binomial(),
                              weights=df['sockets']).fit()

    # Try to compute cluster-robust covariance matrix clustered by specimen.
    # Use statsmodels' sandwich covariance utilities where available.
    cov = None
    try:
        cov = sm.stats.sandwich_covariance.cov_cluster(glm_binom, df['specimen'])
    except Exception:
        # Fallback to heteroskedasticity-robust (HC3) covariance if cluster fails
        try:
            cov = sm.stats.sandwich_covariance.cov_hc3(glm_binom)
        except Exception:
            # Final fallback: use the model's default covariance matrix
            cov = glm_binom.cov_params()

    class RobustResults:
        """
        Lightweight wrapper around a statsmodels results object that uses a
        provided covariance matrix to compute robust standard errors, z-values,
        p-values, and confidence intervals. It preserves access to the original
        results object via the `.base` attribute.
        """
        def __init__(self, base_res, cov_matrix):
            self.base = base_res
            self.cov = cov_matrix
            # ensure parameter order aligns
            self.params = base_res.params
            # Handle case where cov is a DataFrame
            cov_arr = cov_matrix.values if isinstance(cov_matrix, pd.DataFrame) else cov_matrix
            self.bse = np.sqrt(np.maximum(np.diag(cov_arr), 0.0))
            # Avoid divide-by-zero
            with np.errstate(divide='ignore', invalid='ignore'):
                self.tvalues = self.params / self.bse
            # two-sided p-values from normal approximation (GLM large-sample)
            self.pvalues = 2 * (1 - _stats.norm.cdf(np.abs(self.tvalues)))
            self.df_resid = getattr(base_res, 'df_resid', None)

        def conf_int(self, alpha=0.05):
            z = _stats.norm.ppf(1 - alpha / 2)
            ci_lower = self.params - z * self.bse
            ci_upper = self.params + z * self.bse
            return np.column_stack([ci_lower, ci_upper])

        def summary(self):
            # Return a simple pandas DataFrame summarizing coefficients and robust SEs
            ci = self.conf_int()
            summary_df = pd.DataFrame({
                'coef': self.params,
                'std err': self.bse,
                'z': self.tvalues,
                'P>|z|': self.pvalues,
                '[0.025': ci[:, 0],
                '0.975]': ci[:, 1]
            }, index=self.params.index)
            return summary_df

        # Provide attribute access to the underlying results where sensible
        def __getattr__(self, item):
            # Delegate unknown attributes to the base results object
            return getattr(self.base, item)

    robust_res = RobustResults(glm_binom, cov)
    return robust_res