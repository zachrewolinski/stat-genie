from typing import Any, Dict
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools import add_constant
from statsmodels.stats.sandwich_covariance import cov_cluster, cov_hc1
from scipy.stats import norm


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a modeling dataframe. Adds derived variables used in the statistical models:
      - is_female: binary female indicator
      - age_centered, age_squared
      - age_group: coarse categorical bins for descriptive checks
      - MajorityChoice: binary (1 if chose majority option (y==2), 0 otherwise)
      - y_mn: zero-based integer encoding for multinomial model (0,1,2 corresponding to original 1,2,3)

    Drops rows with missing data in key variables used by the models.
    Returns the dataframe containing all columns listed in the conceptual variables.
    """
    df = df.copy()

    # Ensure required columns exist
    required_cols = ['y', 'age', 'culture', 'gender', 'majority_first', 'religiousness', 'school']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"Missing required columns for transform: {missing}")

    # Drop rows missing essential predictors/outcomes
    df = df.dropna(subset=required_cols)

    # Make sure y is integer-coded 1,2,3 (as documented). If not, try coercion.
    df['y'] = df['y'].astype(int)

    # Binary majority choice: 1 if the child selected the majority option (y == 2), else 0
    df['MajorityChoice'] = (df['y'] == 2).astype(int)

    # Create zero-based multinomial outcome for MNLogit (0,1,2 <-> original 1,2,3)
    df['y_mn'] = df['y'].astype(int) - 1

    # Gender: original coding 1 = girl, 2 = boy. Create is_female indicator (1 = girl)
    df['is_female'] = (df['gender'] == 1).astype(int)

    # Age: center and add quadratic term to capture nonlinear development
    df['age'] = df['age'].astype(float)
    df['age_centered'] = df['age'] - df['age'].mean()
    df['age_squared'] = df['age_centered'] ** 2

    # Coarse age groups for descriptive checks (not required by the core models but useful)
    # bins: 4-6, 7-9, 10-14 (adjust boundaries inclusive on the left)
    bins = [3.9, 6.0, 9.0, 14.1]
    labels = ['4-6', '7-9', '10-14']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True)

    # Culture: treat as categorical (keep original numeric codes but as string/category for formulas)
    df['culture'] = df['culture'].astype(str).astype('category')

    # majority_first and religiousness ensure numeric types
    df['majority_first'] = df['majority_first'].astype(int)
    # religiousness may be integer-coded; coerce to numeric
    df['religiousness'] = pd.to_numeric(df['religiousness'], errors='coerce')

    # School: keep as-is for clustering, but ensure no missing and cast to category
    df['school'] = df['school'].astype(str).astype('category')

    # Final drop of rows with NA in any column we will use in the models
    model_cols = [
        'y_mn', 'MajorityChoice', 'age', 'age_centered', 'age_squared', 'age_group',
        'culture', 'is_female', 'majority_first', 'religiousness', 'school'
    ]
    df = df.dropna(subset=model_cols)

    # Reset index for a clean returned dataframe
    df = df.reset_index(drop=True)

    return df


def model(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Runs two complementary statistical models to answer the research question:
      1) A logistic regression (GLM with binomial family) predicting MajorityChoice (binary) to test how reliance on majority information varies with age and culture. The model includes an age x culture interaction to test whether developmental trajectories differ across cultures. Cluster-robust standard errors at the school level are reported.
      2) A multinomial logistic regression (MNLogit) predicting the full 3-way choice (unchosen / majority / minority) with the same predictors (age terms, culture, controls) to examine preference patterns across options. Cluster-robust standard errors at the school level are reported.

    Returns a dictionary with the fitted (robust-cov) results objects for both models.
    """
    results: Dict[str, Any] = {}

    # Ensure the transformed dataframe contains required columns
    required = ['MajorityChoice', 'y_mn', 'age_centered', 'age_squared', 'culture', 'is_female', 'majority_first', 'religiousness', 'school']
    if any(c not in df.columns for c in required):
        raise KeyError(f"Transformed dataframe is missing required columns. Required: {required}")

    # ---------------------------
    # 1) Logistic GLM for MajorityChoice
    # ---------------------------
    # Formula: test main effects of age and culture plus their interaction, controlling for sex, order, religiousness
    formula_glm = 'MajorityChoice ~ age_centered * C(culture) + is_female + majority_first + religiousness'
    glm_model = smf.glm(formula=formula_glm, data=df, family=sm.families.Binomial())
    glm_res = glm_model.fit()

    # Obtain cluster-robust covariance (cluster on school).
    # Some statsmodels result classes expose get_robustcov_results; GLMResults in some versions may not.
    # We handle both cases: prefer get_robustcov_results when available, otherwise compute cluster cov manually
    if hasattr(glm_res, 'get_robustcov_results'):
        try:
            glm_res_clust = glm_res.get_robustcov_results(cov_type='cluster', groups=df['school'])
        except Exception:
            # Fallback: HC1
            glm_res_clust = glm_res.get_robustcov_results(cov_type='HC1')
    else:
        # Compute covariance matrices manually and wrap the original results in a small adapter that exposes commonly used attributes
        try:
            cluster_cov = cov_cluster(glm_res, df['school'])
        except Exception:
            # If cluster fails, fallback to HC1
            cluster_cov = cov_hc1(glm_res)

        class RobustResultWrapper:
            def __init__(self, base_res, cov_matrix):
                self._res = base_res
                self._cov = np.asarray(cov_matrix)
                # params as Series
                self.params = base_res.params
                # compute bse aligned to params.index
                self.bse = pd.Series(np.sqrt(np.diag(self._cov)), index=self.params.index)
                # t-values / z-values
                self.tvalues = self.params / self.bse
                # two-sided p-values using normal approximation
                self.pvalues = 2 * (1 - norm.cdf(np.abs(self.tvalues)))
                # keep original summary available
            def cov_params(self):
                return self._cov
            def summary(self, *args, **kwargs):
                # Return the original summary; it will not reflect cluster SEs, but is available.
                return self._res.summary(*args, **kwargs)
            def __getattr__(self, name):
                # Delegate other attributes to the underlying results object
                return getattr(self._res, name)

        glm_res_clust = RobustResultWrapper(glm_res, cluster_cov)

    results['glm_majority'] = glm_res_clust

    # ---------------------------
    # 2) Multinomial logistic regression for the full 3-way choice
    # ---------------------------
    # Prepare exogenous variables: include age_centered, age_squared, is_female, majority_first, religiousness and culture dummies
    exog_vars = ['age_centered', 'age_squared', 'is_female', 'majority_first', 'religiousness']
    # Create dummies for culture (drop_first=True to avoid multicollinearity); keep names stable
    culture_dummies = pd.get_dummies(df['culture'], prefix='culture', drop_first=True)
    exog = pd.concat([df[exog_vars].reset_index(drop=True), culture_dummies.reset_index(drop=True)], axis=1)
    exog = add_constant(exog, has_constant='add')

    # Endogenous variable must be 0..(J-1)
    endog = df['y_mn'].astype(int)

    mn_model = sm.MNLogit(endog, exog)
    # Fit: increase maxiter in case of slow convergence
    mn_res = mn_model.fit(method='newton', maxiter=200, disp=False)

    # Cluster-robust SEs for MNLogit
    if hasattr(mn_res, 'get_robustcov_results'):
        try:
            mn_res_clust = mn_res.get_robustcov_results(cov_type='cluster', groups=df['school'])
        except Exception:
            mn_res_clust = mn_res.get_robustcov_results(cov_type='HC1')
        results['mnlogit_choice'] = mn_res_clust
    else:
        # If get_robustcov_results is not available, attempt to compute cluster covariance and return the base result,
        # while also storing the cluster covariance matrix for downstream inspection.
        try:
            mn_cluster_cov = cov_cluster(mn_res, df['school'])
        except Exception:
            mn_cluster_cov = cov_hc1(mn_res)
        # Return the original results object and also provide the cluster covariance matrix.
        results['mnlogit_choice'] = mn_res
        results['mnlogit_choice_cluster_cov'] = mn_cluster_cov

    # Also provide some helper objects useful for downstream inspection (design matrices)
    results['mnlogit_exog_names'] = exog.columns.tolist()

    # Return the results dict. Each results entry is a statsmodels Results object (or a wrapper) with .summary() available.
    return results