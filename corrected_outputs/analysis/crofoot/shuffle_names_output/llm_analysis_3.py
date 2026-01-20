from typing import Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import scipy.stats as sps
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle


df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/crofoot/shuffle_names_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Columns used in transformations and modeling
    required_in = ['dyad', 'f_other', 'f_focal', 'win', 'm_focal', 'n_focal', 'other', 'm_other']

    # Drop rows with missing values in essential raw columns
    df = df.dropna(subset=required_in)

    # Interpret distance columns according to the dataset field descriptions:
    # - 'win' (per schema) contains the distance (meters) of the focal group from the center of its home range
    # - 'm_focal' (per schema) contains the distance (meters) of the other group from the center of its home range
    # Create clear named columns for distances in meters
    df['focal_dist_m'] = pd.to_numeric(df['win'], errors='coerce')
    df['other_dist_m'] = pd.to_numeric(df['m_focal'], errors='coerce')

    # Interpret group-size columns according to the provided schema descriptions (these are swapped in the raw file's naming):
    # - 'f_other' (schema) corresponds to the number of individuals in the focal group
    # - 'f_focal' (schema) corresponds to the number of individuals in the other group
    df['focal_group_size'] = pd.to_numeric(df['f_other'], errors='coerce')
    df['other_group_size'] = pd.to_numeric(df['f_focal'], errors='coerce')

    # Ensure male counts are numeric
    df['n_focal'] = pd.to_numeric(df['n_focal'], errors='coerce')
    df['other_males'] = pd.to_numeric(df['other'], errors='coerce')

    # Remove any rows that became NA after coercion
    df = df.dropna(subset=['focal_dist_m', 'other_dist_m', 'focal_group_size', 'other_group_size', 'n_focal', 'other_males', 'dyad', 'm_other'])

    # Outcome variable: ensure binary integer
    df['dyad'] = pd.to_numeric(df['dyad'], errors='coerce').astype(int)

    # Derived predictors
    # Relative group size (focal - other). Positive => focal larger.
    df['RelativeGroupSize'] = df['focal_group_size'] - df['other_group_size']

    # Proximity to focal home: (other_dist_m - focal_dist_m). Positive => contest is relatively closer to focal group's home.
    df['ProximityToFocalHome'] = df['other_dist_m'] - df['focal_dist_m']

    # Standardize (z-score) the two main continuous predictors for better numeric behavior and interaction interpretation
    df['RelativeGroupSize_z'] = (df['RelativeGroupSize'] - df['RelativeGroupSize'].mean()) / (df['RelativeGroupSize'].std(ddof=0) if df['RelativeGroupSize'].std(ddof=0) != 0 else 1)
    df['ProximityToFocalHome_z'] = (df['ProximityToFocalHome'] - df['ProximityToFocalHome'].mean()) / (df['ProximityToFocalHome'].std(ddof=0) if df['ProximityToFocalHome'].std(ddof=0) != 0 else 1)

    # Keep only columns needed for modeling (but retain original columns as well)
    model_cols = ['dyad', 'RelativeGroupSize', 'RelativeGroupSize_z', 'ProximityToFocalHome', 'ProximityToFocalHome_z', 'n_focal', 'other_males', 'm_other', 'focal_group_size', 'other_group_size', 'focal_dist_m', 'other_dist_m']
    # Some downstream code may still expect original columns; return full df but ensure model columns exist and have no NA
    df = df.loc[df[model_cols].notnull().all(axis=1)].copy()

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Formula: main effects of relative group size and contest location, plus their interaction.
    # Control for numbers of adult males in each group. Use clustered standard errors by dyad pair (m_other).
    formula = 'dyad ~ RelativeGroupSize_z * ProximityToFocalHome_z + n_focal + other_males'

    # Fit a binomial GLM (logistic regression)
    glm_res = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Obtain cluster-robust covariance (cluster on dyad / pair id 'm_other')
    # Compute clustered covariance matrix using statsmodels' sandwich_covariance utility
    from statsmodels.stats.sandwich_covariance import cov_cluster

    # Ensure groups is an array-like with same length as the model's endog
    groups = df['m_other'].values
    clustered_cov = cov_cluster(glm_res, groups)

    # Build a lightweight results-like object that exposes params, bse, pvalues, conf_int, and summary
    class ClusteredResults:
        def __init__(self, base_res, cov):
            self._base = base_res
            self.params = base_res.params
            self.cov_params = lambda: cov
            self.bse = pd.Series(np.sqrt(np.diag(cov)), index=self.params.index)
            # z-statistics and two-sided p-values (normal approximation)
            self.zvalues = self.params / self.bse
            self.pvalues = pd.Series(2 * sps.norm.sf(np.abs(self.zvalues)), index=self.params.index)

        def conf_int(self, alpha=0.05):
            zcrit = sps.norm.ppf(1 - alpha / 2)
            lower = self.params - zcrit * self.bse
            upper = self.params + zcrit * self.bse
            ci = pd.DataFrame({0: lower, 1: upper})
            return ci

        def summary(self):
            # Return the base summary as a fallback (may reflect original cov), but include note about clustered cov
            try:
                base_summary = self._base.summary()
            except Exception:
                base_summary = str(self._base)
            return base_summary

    clustered_res = ClusteredResults(glm_res, clustered_cov)

    return clustered_res