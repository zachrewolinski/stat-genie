from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe suitable for binomial GLM analysis of AMTL.

    Outputs (columns required for modeling):
      - num_amtl (int): number of antemortem missing teeth (per observation)
      - sockets (int): number of observable sockets (trials)
      - amtl_prop (float): proportion num_amtl / sockets
      - genus (category): cleaned genus label
      - age (float): estimated age at death
      - prob_male (float): estimated probability male
      - tooth_class (category): tooth class (Anterior, Posterior, Premolar)
      - specimen (category): specimen id (for clustering)

    Notes:
      - Rows with sockets <= 0 or missing critical fields are removed.
      - Values where num_amtl > sockets are clipped to sockets.
    """
    df = df.copy()

    # Standardize column names (in case of stray whitespace)
    df.columns = [c.strip() for c in df.columns]

    # Drop rows missing required variables for modeling
    required = ['num_amtl', 'sockets', 'genus', 'age', 'prob_male', 'tooth_class', 'specimen']
    df = df.dropna(subset=required)

    # Ensure numeric types for counts
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df = df.dropna(subset=['sockets', 'num_amtl'])

    # Remove rows where sockets <= 0 (can't model binomial proportion)
    df = df[df['sockets'] > 0]

    # If num_amtl > sockets (data error), clip to sockets
    mask_over = df['num_amtl'] > df['sockets']
    if mask_over.any():
        df.loc[mask_over, 'num_amtl'] = df.loc[mask_over, 'sockets']

    # Create proportion column for use with binomial family and frequency weights
    df['amtl_prop'] = df['num_amtl'] / df['sockets']

    # Clean genus labels (strip whitespace); ensure consistent naming for Homo sapiens
    df['genus'] = df['genus'].astype(str).str.strip()
    # common normalization (map short forms if present)
    df['genus'] = df['genus'].replace({
        'Homo': 'Homo sapiens',
        'Homo_sapiens': 'Homo sapiens',
        'Homo sapiens ': 'Homo sapiens'
    })

    # Convert tooth_class and specimen to categorical
    df['tooth_class'] = df['tooth_class'].astype('category')
    df['specimen'] = df['specimen'].astype('category')
    df['genus'] = df['genus'].astype('category')

    # Ensure age and prob_male are numeric
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')

    # Drop any rows with newly introduced NA in key covariates
    df = df.dropna(subset=['age', 'prob_male'])

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial GLM (logit link) modeling AMTL proportion with trials = sockets.

    Model formula:
      amtl_prop ~ C(genus, Treatment(reference="Homo sapiens")) + age + prob_male + C(tooth_class)

    The model is fitted with frequency weights = sockets so that the response is treated as num_amtl out of sockets.
    Robust standard errors clustered by specimen are returned to account for non-independence of observations from the same specimen.

    Returns:
      - A dict containing:
        - 'glm_result_clustered': a lightweight wrapper exposing clustered covariance and params
        - 'odds_ratio_table': DataFrame of ORs, 95% CI, coefficients, clustered SEs, and p-values
        - 'genus_coefficients': dict of genus-related coefficients
        - 'formula': the formula string used
    """
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm  # ensure families available
    import statsmodels.formula.api as smf
    from statsmodels.stats.sandwich_covariance import cov_cluster
    from scipy.stats import norm

    # Ensure required columns exist
    required = ['amtl_prop', 'num_amtl', 'sockets', 'genus', 'age', 'prob_male', 'tooth_class', 'specimen']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Dataframe is missing required columns for modeling: {missing}")

    # Fit GLM using proportion response with frequency weights = sockets (binomial)
    # Use Homo sapiens as reference level for genus via Treatment coding in patsy
    formula = 'amtl_prop ~ C(genus, Treatment(reference="Homo sapiens")) + age + prob_male + C(tooth_class)'

    # Fit the model
    glm_model = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), freq_weights=df['sockets'])
    res = glm_model.fit()

    # Compute clustered covariance matrix using specimen as clusters
    # cov_cluster expects the results instance and group labels
    groups = df['specimen']
    cov = cov_cluster(res, groups)

    # Compute clustered standard errors, z-stats, and p-values (normal approximation)
    params = res.params
    se = np.sqrt(np.diag(cov))
    # ensure alignment as Series
    se = pd.Series(se, index=params.index)
    z_scores = params / se
    pvalues = 2 * (1 - norm.cdf(np.abs(z_scores)))

    # Compute odds ratios and 95% CIs using clustered covariance
    or_vals = np.exp(params)
    lower = np.exp(params - 1.96 * se)
    upper = np.exp(params + 1.96 * se)

    or_table = pd.DataFrame({
        'OR': or_vals,
        'OR_95ci_lower': lower,
        'OR_95ci_upper': upper,
        'coef': params,
        'se_clustered': se,
        'pvalue': pvalues
    })

    # For interpretability: extract genus coefficients (differences relative to Homo sapiens).
    genus_coefs = {k: v for k, v in params.items() if 'genus' in k}

    # Create a lightweight wrapper that exposes clustered covariance and associated info,
    # while delegating other attributes to the original results object.
    class ClusteredResults:
        def __init__(self, base_res, clustered_cov, clustered_pvalues):
            self._base = base_res
            self.params = base_res.params
            self._clustered_cov = pd.DataFrame(clustered_cov, index=base_res.params.index, columns=base_res.params.index)
            self.pvalues = pd.Series(clustered_pvalues, index=base_res.params.index)

        def cov_params(self):
            return self._clustered_cov

        def summary(self):
            # Return base summary; covariance shown won't reflect clustered cov, but key items are accessible
            return self._base.summary()

        def __getattr__(self, name):
            # Delegate attribute access to the underlying results object
            return getattr(self._base, name)

    res_clust = ClusteredResults(res, cov, pvalues)

    results = {
        'glm_result_clustered': res_clust,
        'odds_ratio_table': or_table,
        'genus_coefficients': genus_coefs,
        'formula': formula
    }

    return results