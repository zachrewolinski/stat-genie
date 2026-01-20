from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster
from scipy import stats


df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/amtl/noperturb_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Gilmore (2013) AMTL dataset into the form required for binomial regression.

    Produces the following new/ensured columns used in the model:
      - num_amtl (int) : number of missing teeth in the given tooth_class
      - sockets (int)  : number of observable sockets (trials)
      - prop_amtl (float) : proportion missing = num_amtl / sockets
      - age_c (float) : mean-centered age
      - prob_male_c (float) : mean-centered prob_male
      - genus (str/categorical) : genus string (kept as-is)
      - tooth_class (str/categorical) : tooth class string
      - specimen (str) : specimen identifier (for clustering)

    Rows with missing or invalid trial counts (sockets <= 0) or missing required fields are dropped.
    """
    df = df.copy()

    # Required columns
    required = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen']
    df = df.dropna(subset=required)

    # Ensure numeric types for counts/trials
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')

    # Drop rows with non-positive or missing sockets/num_amtl
    df = df.dropna(subset=['sockets', 'num_amtl'])
    df = df[df['sockets'] > 0]

    # Round/convert counts to integers and ensure 0 <= num_amtl <= sockets
    df['sockets'] = df['sockets'].round().astype(int)
    df['num_amtl'] = df['num_amtl'].round().astype(int)
    # Clip num_amtl to valid range per row
    df['num_amtl'] = df.apply(lambda row: int(max(0, min(int(row['num_amtl']), int(row['sockets'])))), axis=1)

    # Keep only genera expected in the study to avoid unexpected categories
    allowed_genera = {'Homo sapiens', 'Pan', 'Pongo', 'Papio'}
    df = df[df['genus'].isin(allowed_genera)]

    # Compute proportion (for use with Binomial family using weights = sockets)
    df['prop_amtl'] = df['num_amtl'] / df['sockets']

    # Center continuous controls to aid interpretation / numerical stability
    df['age_c'] = df['age'] - df['age'].mean()
    df['prob_male_c'] = df['prob_male'] - df['prob_male'].mean()

    # Ensure categorical columns are strings (patsy/statsmodels handles categorical conversion)
    df['genus'] = df['genus'].astype(str)
    df['tooth_class'] = df['tooth_class'].astype(str)
    df['specimen'] = df['specimen'].astype(str)

    # Final check: drop any rows with missing values in model columns
    model_cols = ['prop_amtl', 'num_amtl', 'sockets', 'genus', 'age_c', 'prob_male_c', 'tooth_class', 'specimen']
    df = df.dropna(subset=model_cols)

    # Return only columns needed for modeling plus original counts for clarity
    keep_cols = ['num_amtl', 'sockets', 'prop_amtl', 'genus', 'age_c', 'prob_male_c', 'tooth_class', 'specimen']
    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial GLM to test whether Homo sapiens have higher AMTL than non-human primates,
    controlling for age, sex (probability male), and tooth class.

    Model specification (using proportion + weights):
      prop_amtl ~ C(genus, Treatment(reference="Homo sapiens")) + age_c + prob_male_c + C(tooth_class)
    Family: Binomial
    Weights: sockets (number of trials)

    Returns an object exposing .params, .bse, .pvalues, .cov_params(), and .summary().
    The .bse/.cov_params()/.pvalues are computed using cluster-robust standard errors clustered by specimen.
    """
    # Formula: compare genera with 'Homo sapiens' as reference
    formula = 'prop_amtl ~ C(genus, Treatment(reference="Homo sapiens")) + age_c + prob_male_c + C(tooth_class)'

    # Fit GLM with binomial family using sockets as frequency weights (trials)
    glm_mod = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), freq_weights=df['sockets'])
    res = glm_mod.fit()

    # Compute cluster-robust covariance matrix clustered by specimen
    groups = df['specimen'].values
    cluster_cov = cov_cluster(res, groups)

    class ClusterRobustResults:
        def __init__(self, original_res, cluster_cov_matrix, cluster_groups):
            self._res = original_res
            self._cov = cluster_cov_matrix
            self._groups = cluster_groups
            # ensure ordering aligns with params
            self.params = self._res.params
            self.bse = np.sqrt(np.diag(self._cov))
            # p-values using normal approximation (Wald z)
            zvals = self.params.values / self.bse
            self.pvalues = pd.Series(2 * stats.norm.sf(np.abs(zvals)), index=self.params.index)
            # store cov as DataFrame for easier inspection with proper index/columns
            try:
                self.cov_params = pd.DataFrame(self._cov, index=self.params.index, columns=self.params.index)
            except Exception:
                # fallback: raw array
                self.cov_params = self._cov

        def cov_params_default(self):
            return self._cov

        def summary(self):
            # Return the original summary; users can inspect .params/.bse/.pvalues for cluster-robust stats
            return self._res.summary()

        # allow attribute access to original result for other methods/attributes if needed
        def __getattr__(self, item):
            return getattr(self._res, item)

    robust_res = ClusterRobustResults(res, cluster_cov, groups)
    return robust_res