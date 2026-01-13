from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster
import scipy.stats as stats

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/replace_with_rvs_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for modeling.

    Produces the following new columns used in the model:
      - MajorityChosen : int (1 if y==2 [majority], else 0)
      - gender_male    : int (1 if gender==2 (boy), 0 if gender==1 (girl))
      - age_c          : centered age (age - mean(age))
      - age_c2         : squared centered age
      - majority_first : integer (kept from input, ensures 0/1)
      - culture        : integer site identifier (kept as int for categorical coding in model)

    Drops rows with missing values on required columns.
    """
    # work on a copy
    df = df.copy()

    # Required columns
    req_cols = ['y', 'age', 'gender', 'majority_first', 'culture']
    # Drop rows missing any required variable
    df = df.dropna(subset=req_cols)

    # Dependent variable: majority chosen (y==2 means majority option)
    df['MajorityChosen'] = (df['y'] == 2).astype(int)

    # Gender encoding: input uses 1=girl, 2=boy -> create gender_male as 1 for boy
    df['gender_male'] = df['gender'].map({1: 0, 2: 1})
    # If any unexpected values remain (not 1 or 2), coerce to NaN and drop
    if df['gender_male'].isnull().any():
        df = df.dropna(subset=['gender_male'])
    df['gender_male'] = df['gender_male'].astype(int)

    # Ensure majority_first is integer 0/1
    df['majority_first'] = df['majority_first'].astype(int)

    # Culture as integer categorical indicator
    df['culture'] = df['culture'].astype(int)

    # Center age and add quadratic term to capture nonlinear development
    df['age_c'] = df['age'] - df['age'].mean()
    df['age_c2'] = df['age_c'] ** 2

    # Final check: drop any rows with NA in columns we will use in modeling
    model_cols = ['MajorityChosen', 'age_c', 'age_c2', 'culture', 'gender_male', 'majority_first']
    df = df.dropna(subset=model_cols)

    # Return transformed dataframe with all columns needed for modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression predicting the probability that a child chooses the majority option.

    Model specification (primary):
      MajorityChosen ~ age_c * C(culture) + age_c2 + gender_male + majority_first

    - age_c * C(culture) tests whether age-related change in majority reliance differs across cultures (interaction).
    - age_c2 captures potential nonlinearity in the age effect.
    - gender_male and majority_first are included as control variables.

    Because observations are clustered by culture/site, cluster-robust standard errors (cluster = culture)
    are estimated.

    Returns a results-like object that exposes params, bse, pvalues, conf_int, cov (cluster-robust covariance),
    and a summary() method.
    """
    # Work on a copy to avoid modifying the input
    df = df.copy()

    # Define formula: include culture as categorical factor and interact with centered age
    formula = 'MajorityChosen ~ age_c * C(culture) + age_c2 + gender_male + majority_first'

    # Fit a standard logistic regression (maximum likelihood)
    logit_model = smf.logit(formula=formula, data=df)
    fitted = logit_model.fit(disp=False)

    # Obtain cluster-robust covariance (clusters = culture) using statsmodels utility
    cluster_groups = df['culture'].astype(int).values
    # cov_cluster accepts the fitted results object and a group vector
    cluster_cov = cov_cluster(fitted, cluster_groups)

    # Compute robust standard errors, z-stats, p-values, and confidence intervals
    params = fitted.params
    bse = np.sqrt(np.diag(cluster_cov))
    # Protect against zero or negative variances (shouldn't happen, but guard)
    bse = np.where(bse == 0, np.nan, bse)
    z_stats = params / bse
    pvalues = 2 * (1 - stats.norm.cdf(np.abs(z_stats)))

    def conf_int(alpha=0.05):
        crit = stats.norm.ppf(1 - alpha / 2)
        lower = params - crit * bse
        upper = params + crit * bse
        return np.column_stack((lower, upper))

    # Create a simple results-like wrapper to hold cluster-robust inference
    class ClusterRobustResults:
        def __init__(self, orig, cov, params, bse, pvalues, conf_int_func):
            self._orig = orig
            self.cov = cov
            self.params = params
            self.bse = bse
            self.pvalues = pvalues
            self._conf_int_func = conf_int_func
            # expose some original attributes for compatibility
            self.model = getattr(orig, 'model', None)
            self.df_model = getattr(orig, 'df_model', None)
            self.df_resid = getattr(orig, 'df_resid', None)

        def cov_params(self):
            return self.cov

        def conf_int(self, alpha=0.05):
            return self._conf_int_func(alpha)

        def summary(self):
            """
            Attempt to return the original summary; if that is not informative about cluster-robust SEs,
            provide a concise table using cluster-robust estimates.
            """
            try:
                # try to display the original summary (may not reflect cluster-robust SEs)
                orig_summary = self._orig.summary()
            except Exception:
                orig_summary = None

            # Build a concise DataFrame with cluster-robust stats
            try:
                import pandas as _pd
                ci = self.conf_int()
                table = _pd.DataFrame({
                    'coef': self.params,
                    'std err (cluster)': self.bse,
                    'z': self.params / self.bse,
                    'P>|z|': self.pvalues,
                    '[0.025': ci[:, 0],
                    '0.975]': ci[:, 1]
                })
                if orig_summary is not None:
                    return orig_summary
                return table
            except Exception:
                # As a final fallback, return a string
                return f'params:\n{self.params}\n\nbse (cluster):\n{self.bse}\n\npvalues:\n{self.pvalues}'

    results = ClusterRobustResults(fitted, cluster_cov, params, bse, pvalues, conf_int)

    # Print a brief summary (caller can inspect returned results as well)
    try:
        print(results.summary())
    except Exception:
        # In some runtime contexts printing the summary may fail; ignore silently
        pass

    return results