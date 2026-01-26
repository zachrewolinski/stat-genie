from typing import Any, Dict
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm


df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/replace_with_rvs_output/amtl.csv')


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw AMTL dataset into analysis-ready dataframe.

    Produces the following new/ensured columns used in the model:
      - num_amtl (int): number of missing teeth in the tooth_class for that specimen (kept as provided, capped at sockets)
      - sockets (int): number of observable sockets (trials)
      - amtl_prop (float): num_amtl / sockets (proportion of missing teeth in that class)
      - is_human (int): 1 if genus == 'Homo sapiens', else 0
      - age_c (float): centered age (age - mean(age))
      - tooth_class (category): category ('Anterior','Posterior','Premolar')
      - prob_male (float): kept as provided (0-1)
      - specimen (category): specimen id (kept)

    Rows with missing critical data or invalid sockets are dropped.
    """
    # Work on a copy
    df = df.copy()

    # Required columns existence check (will raise if missing)
    required_cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Drop rows with NA in critical columns
    df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen'])

    # Ensure numeric types where relevant
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')

    # Remove rows with invalid sockets (must be >=1) or missing after coercion
    df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male'])
    df = df[df['sockets'] >= 1]

    # Cap num_amtl at sockets if any data errors (cannot have more successes than trials)
    df.loc[df['num_amtl'] > df['sockets'], 'num_amtl'] = df.loc[df['num_amtl'] > df['sockets'], 'sockets']
    df.loc[df['num_amtl'] < 0, 'num_amtl'] = 0

    # Ensure num_amtl is integer-valued (counts)
    # Use floor of values after capping to be safe if non-integers were present
    df['num_amtl'] = np.floor(df['num_amtl']).astype(int)
    df['sockets'] = np.floor(df['sockets']).astype(int)

    # Create proportion outcome for binomial modeling
    df['amtl_prop'] = df['num_amtl'] / df['sockets']

    # Create binary indicator for human vs non-human
    # Keep exact string matching for 'Homo sapiens' as indicated in dataset description
    df['is_human'] = (df['genus'].astype(str).str.strip() == 'Homo sapiens').astype(int)

    # Center age for interpretability
    df['age_c'] = df['age'] - df['age'].mean()

    # Ensure tooth_class is a categorical with expected levels (preserve existing if possible)
    df['tooth_class'] = df['tooth_class'].astype('category')

    # Ensure specimen is categorical (used for clustering)
    df['specimen'] = df['specimen'].astype('category')

    # Keep only the columns needed for modeling (but don't drop others unnecessarily)
    required_for_model = ['num_amtl', 'sockets', 'amtl_prop', 'is_human', 'age_c', 'prob_male', 'tooth_class', 'specimen', 'genus']
    df = df.loc[:, [c for c in required_for_model if c in df.columns]]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Fit a binomial GLM (logit link) predicting proportion of AMTL with robust clustered SE by specimen.

    Model specification:
      Response: amtl_prop (num_amtl / sockets) with weights = sockets (number of trials)
      Predictors: is_human (primary IV), age_c (centered age), prob_male (sex probability), C(tooth_class)

    Returns a dictionary containing:
      - fit: the original GLM fit result (statsmodels object)
      - robust_results: fit with cluster-robust covariance (clustered by specimen) wrapped object
      - summary: robust summary text
      - odds_ratios: point estimates (exp(coef)) as a pandas Series
      - conf_int_or: odds-ratio confidence intervals as a DataFrame
    """
    # Check required columns
    for col in ['num_amtl', 'sockets', 'amtl_prop', 'is_human', 'age_c', 'prob_male', 'tooth_class', 'specimen']:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe passed to model(). Run transform() first.")

    # Ensure no zero-division or invalid proportions
    if (df['sockets'] <= 0).any():
        raise ValueError('All sockets must be >= 1 for binomial modeling with weights.')

    # Prepare a copy for modeling and create a clipped proportion to avoid boundary issues
    df_model = df.copy()

    # Clip amtl_prop away from exact 0 or 1 to avoid numerical issues in fitting
    eps = 1e-6
    df_model['amtl_prop_model'] = df_model['amtl_prop'].clip(eps, 1 - eps)

    # Build formula: use the clipped proportion as the endogenous variable and sockets as weights
    formula = 'amtl_prop_model ~ is_human + age_c + prob_male + C(tooth_class)'

    # Fit GLM with binomial family; weights = sockets (number of trials)
    glm_mod = smf.glm(formula=formula, data=df_model, family=sm.families.Binomial(), weights=df_model['sockets'])
    fit = glm_mod.fit()

    # Compute cluster-robust covariance matrix clustered by specimen.
    # Use statsmodels' sandwich covariance utility to compute clustered covariance.
    # We avoid relying on a get_robustcov_results method that may not exist in some versions.
    try:
        # Preferred: use statsmodels built-in sandwich_covariance.cov_cluster
        cov_cluster = sm.stats.sandwich_covariance.cov_cluster(fit, df_model['specimen'].values)
    except Exception:
        # Fallback: attempt to compute using alternative path
        cov_cluster = sm.stats.sandwich_covariance.cov_cluster(fit, df_model['specimen'].values)

    class _SummaryText:
        def __init__(self, text: str):
            self._text = text

        def as_text(self) -> str:
            return self._text

    class RobustResultsWrapper:
        """
        Minimal wrapper to expose params, conf_int, and a summary-like object
        while using the cluster-robust covariance matrix computed above.
        """
        def __init__(self, fit_res, cov):
            self._fit = fit_res
            self._cov = cov
            # params are the same point estimates as the original fit
            self.params = pd.Series(self._fit.params, index=self._fit.params.index)

        @property
        def bse(self) -> pd.Series:
            se = np.sqrt(np.diag(self._cov))
            return pd.Series(se, index=self.params.index)

        def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
            z = norm.ppf(1 - alpha / 2)
            se = self.bse
            lower = self.params - z * se
            upper = self.params + z * se
            return pd.DataFrame({0: lower, 1: upper}, index=self.params.index)

        def summary(self):
            # Provide the original fit summary text and note that covariances were replaced
            base = ""
            try:
                base = self._fit.summary().as_text()
            except Exception:
                # If summary() fails for some reason, provide basic info
                base = f"Coefficients:\n{self.params.to_string()}\n"
            note = "\nNote: covariance matrix replaced by cluster-robust covariance clustered on 'specimen'."
            return _SummaryText(base + note)

    robust = RobustResultsWrapper(fit, cov_cluster)

    # Prepare summary and effect size (odds ratios)
    summary_text = robust.summary().as_text()

    coef = robust.params
    conf_int = robust.conf_int()

    # Odds ratios and corresponding CIs
    odds_ratios = np.exp(coef)
    conf_int_or = np.exp(conf_int)

    # Format results into pandas objects for convenient downstream use
    odds_ratios = pd.Series(odds_ratios, index=coef.index, name='odds_ratio')
    conf_int_or = pd.DataFrame(conf_int_or, index=coef.index, columns=['ci_lower_or', 'ci_upper_or'])

    results = {
        'fit': fit,
        'robust_results': robust,
        'summary': summary_text,
        'odds_ratios': odds_ratios,
        'conf_int_or': conf_int_or
    }

    return results