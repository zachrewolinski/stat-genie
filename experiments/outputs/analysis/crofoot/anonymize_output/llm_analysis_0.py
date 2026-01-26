from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster, cov_hc1
from scipy import stats as sps


df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/anonymize_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a modeling-ready dataframe. Returns a dataframe containing the columns
    used in the statistical model and supporting diagnostics.

    Expected input columns (original names):
      - feature1: focal group ID
      - feature2: other group ID
      - feature3: dyad ID
      - feature4: 1 if focal won, 0 if other won
      - feature5: focal distance from its home-range center (meters)
      - feature6: other distance from its home-range center (meters)
      - feature7: focal group size (n individuals)
      - feature8: other group size
      - feature9: focal number of males
      - feature10: other number of males
      - feature11: focal number of females
      - feature12: other number of females
    """
    df = df.copy()

    # 1) Rename columns to meaningful names used downstream
    rename_map = {
        'feature1': 'FocalID',
        'feature2': 'OtherID',
        'feature3': 'DyadID',
        'feature4': 'FocalWon',
        'feature5': 'FocalDistHome',
        'feature6': 'OtherDistHome',
        'feature7': 'FocalGroupSize',
        'feature8': 'OtherGroupSize',
        'feature9': 'FocalMales',
        'feature10': 'OtherMales',
        'feature11': 'FocalFemales',
        'feature12': 'OtherFemales'
    }
    df = df.rename(columns=rename_map)

    # 2) Keep only rows with required values and coerce numeric types
    required = ['FocalWon', 'FocalDistHome', 'OtherDistHome', 'FocalGroupSize', 'OtherGroupSize', 'FocalMales', 'OtherMales', 'FocalFemales', 'OtherFemales', 'DyadID']
    df = df.dropna(subset=required).copy()

    numeric_cols = ['FocalWon', 'FocalDistHome', 'OtherDistHome', 'FocalGroupSize', 'OtherGroupSize', 'FocalMales', 'OtherMales', 'FocalFemales', 'OtherFemales']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # drop any rows that became NA after coercion
    df = df.dropna(subset=numeric_cols + ['DyadID']).copy()

    # Ensure binary outcome is integer 0/1
    # Clip values to 0/1 just in case and convert to int
    df['FocalWon'] = df['FocalWon'].astype(float).round().clip(lower=0, upper=1).astype(int)

    # 3) Construct predictor variables
    # Relative size (difference) and also a log ratio (useful for robustness checks)
    df['RelativeSize'] = df['FocalGroupSize'] - df['OtherGroupSize']
    df['SizeLogRatio'] = np.log((df['FocalGroupSize'] + 1) / (df['OtherGroupSize'] + 1))

    # Relative sex-composition
    df['RelativeMales'] = df['FocalMales'] - df['OtherMales']
    df['RelativeFemales'] = df['FocalFemales'] - df['OtherFemales']

    # 4) Derive contest location relative to home-range centers
    # Positive DistanceDiff means focal is closer to its home center than other (other is farther)
    df['DistanceDiff'] = df['OtherDistHome'] - df['FocalDistHome']

    # threshold for being 'substantially closer' (in meters). This is a tunable parameter.
    threshold = 50
    def classify_location(diff):
        # diff = OtherDistHome - FocalDistHome
        if diff > threshold:
            return 'FocalHome'   # other is much farther -> focal is nearer its home
        elif diff < -threshold:
            return 'OtherHome'   # focal is much farther -> other is nearer its home
        else:
            return 'Neutral'

    df['ContestLocation'] = df['DistanceDiff'].apply(classify_location).astype('category')

    # 5) Convert DyadID to a categorical-like identifier (keep original values too)
    # Ensure DyadID is a stable grouping variable (string) for clustering
    df['DyadID'] = df['DyadID'].astype(str)

    # 6) Optional: drop rows with extremely imbalanced or impossible values (none expected here)
    # (No further filtering by default.)

    # 7) Final column list used by the model. Keep them in the dataframe for inspection.
    final_cols = ['FocalID', 'OtherID', 'DyadID', 'FocalWon', 'FocalDistHome', 'OtherDistHome', 'DistanceDiff',
                  'ContestLocation', 'FocalGroupSize', 'OtherGroupSize', 'RelativeSize', 'SizeLogRatio',
                  'FocalMales', 'OtherMales', 'RelativeMales', 'FocalFemales', 'OtherFemales', 'RelativeFemales']

    # Ensure final columns exist (some may be missing in pathological inputs)
    existing_final_cols = [c for c in final_cols if c in df.columns]
    return df[existing_final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression predicting focal group win (binary) from relative group size,
    contest location, and controls. Returns a results-like object with cluster-robust SEs by DyadID.

    Model specification (primary):
      FocalWon ~ RelativeSize * C(ContestLocation) + RelativeMales + RelativeFemales

    Notes:
      - C(ContestLocation) treats the location as a categorical factor with baseline 'Neutral'.
      - Interaction tests whether relative size effects differ depending on location (home advantage).
      - Clustered SEs by DyadID account for repeated observations of the same dyad.
    """
    df = df.copy()

    # Ensure variables have correct types
    df['ContestLocation'] = df['ContestLocation'].astype('category')
    df['DyadID'] = df['DyadID'].astype(str)

    # Primary formula with interaction between relative size and contest location
    formula = 'FocalWon ~ RelativeSize * C(ContestLocation) + RelativeMales + RelativeFemales'

    # Fit logistic regression (maximum likelihood)
    model_fit = smf.logit(formula=formula, data=df)
    try:
        res = model_fit.fit(disp=False)
    except Exception as e:
        # If the model fails to converge, return the error for debugging
        raise RuntimeError(f"Logit fit failed: {e}")

    # Compute clustered covariance (cluster on DyadID). Fallback to HC1 if clustering fails.
    try:
        cov = cov_cluster(res, df['DyadID'].values)
    except Exception:
        cov = cov_hc1(res)

    # Construct a lightweight results-like wrapper that exposes common attributes
    class RobustResults:
        def __init__(self, orig_res, cov_mat):
            self.orig_res = orig_res
            self.params = orig_res.params
            self.cov_params = cov_mat
            self.bse = np.sqrt(np.diag(cov_mat))
            # Use normal approximation for z-stats
            self.zvalues = self.params / self.bse
            self.pvalues = 2 * (1 - sps.norm.cdf(np.abs(self.zvalues)))
            # 95% CI
            crit = sps.norm.ppf(0.975)
            lower = self.params - crit * self.bse
            upper = self.params + crit * self.bse
            self.conf_int = np.vstack([lower, upper]).T

        def summary(self):
            # Build a summary table similar to statsmodels' param table
            table = pd.DataFrame({
                'coef': self.params,
                'std err': self.bse,
                'z': self.zvalues,
                'P>|z|': self.pvalues,
                '[0.025': self.conf_int[:, 0],
                '0.975]': self.conf_int[:, 1]
            })
            print(table.to_string(float_format=lambda x: f"{x:0.4f}"))

        # Allow attribute access to original result for other details if needed
        def __getattr__(self, name):
            return getattr(self.orig_res, name)

    res_cluster = RobustResults(res, cov)

    # Print a concise summary (users can inspect the returned object for full details)
    res_cluster.summary()

    return res_cluster