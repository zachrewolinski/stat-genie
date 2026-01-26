from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster
from scipy.stats import norm


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw contest dataframe to create derived predictors and standardized controls used in modeling.

    Produces the following new columns (all included in the returned dataframe):
      - size_diff: n_focal - n_other
      - size_ratio: n_focal / n_other
      - size_adv: binary indicator if focal group larger than other
      - rel_distance: dist_other - dist_focal (positive => focal is closer to its home center)
      - location_adv: binary indicator if focal is closer to its own home center than other
      - m_diff: m_focal - m_other
      - f_diff: f_focal - f_other
      - z_size_diff, z_rel_distance, z_m_diff, z_f_diff: standardized versions (mean 0, sd 1)
      - focal and dyad converted to categorical types

    Rows with missing values on key fields (win, group sizes, distances, sex counts) are dropped.
    """
    df = df.copy()

    # Drop rows missing essential columns (cannot evaluate contest outcome predictors without these)
    required = [
        'win',
        'n_focal', 'n_other',
        'dist_focal', 'dist_other',
        'm_focal', 'm_other',
        'f_focal', 'f_other',
        'dyad', 'focal'
    ]
    df = df.dropna(subset=required)

    # Derived size metrics
    df['size_diff'] = df['n_focal'] - df['n_other']
    # Guard against division by zero
    df['size_ratio'] = df['n_focal'] / df['n_other'].replace({0: np.nan})
    df['size_adv'] = (df['size_diff'] > 0).astype(int)

    # Location / proximity metrics: positive rel_distance means focal is relatively closer to its home center
    df['rel_distance'] = df['dist_other'] - df['dist_focal']
    df['location_adv'] = (df['dist_focal'] < df['dist_other']).astype(int)

    # Sex-composition differences as potential controls
    df['m_diff'] = df['m_focal'] - df['m_other']
    df['f_diff'] = df['f_focal'] - df['f_other']

    # Standardize continuous predictor/control variables for interpretability
    for col in ['size_diff', 'rel_distance', 'm_diff', 'f_diff']:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            df['z_' + col] = 0.0
        else:
            df['z_' + col] = (df[col] - mean) / std

    # Ensure categorical columns have appropriate dtype for modeling
    df['focal'] = df['focal'].astype('category')
    df['dyad'] = df['dyad'].astype('category')

    # Return the full transformed dataframe so users can inspect intermediate variables if desired
    return df


class ClusteredResults:
    """
    Lightweight container for clustered covariance-adjusted inference results.

    Provides a .summary() method that prints a table similar to statsmodels' summary for coefficients,
    but uses the supplied clustered covariance matrix for standard errors.
    """

    def __init__(self, model_result, cov, name: str = None):
        self.model_result = model_result
        self.cov_params = cov
        self.params = model_result.params
        self.bse = np.sqrt(np.diag(self.cov_params))
        # Use normal z-statistics (large-sample approximation commonly used for logit)
        self.zvalues = self.params / self.bse
        self.pvalues = 2 * (1 - norm.cdf(np.abs(self.zvalues)))
        self.conf_int = np.vstack([
            self.params - 1.96 * self.bse,
            self.params + 1.96 * self.bse
        ]).T
        self._name = name or getattr(model_result.model, 'endog_names', 'model')

    def summary(self) -> str:
        df = pd.DataFrame({
            'coef': self.params,
            'std err': self.bse,
            'z': self.zvalues,
            'P>|z|': self.pvalues,
            '[0.025': self.conf_int[:, 0],
            '0.975]': self.conf_int[:, 1]
        })
        df.index.name = 'coef'
        return f"Clustered results ({self._name}):\n" + df.to_string(float_format=lambda x: f"{x:.4f}")


def _make_clustered_results(result, groups, name: str = None) -> ClusteredResults:
    """
    Given a fitted statsmodels result object and a group series/array,
    compute clustered covariance and return a ClusteredResults wrapper.
    """
    # cov_cluster expects either the result or exog and resid; passing result is supported
    cov = cov_cluster(result, groups)
    return ClusteredResults(result, cov, name=name)


def model(df: pd.DataFrame) -> Any:
    """
    Fit logistic regression models to estimate the effect of relative group size and contest location on probability of focal group winning.

    Models fitted:
      1) Primary logit with interaction between standardized size difference and standardized relative distance, controlling for sex-composition differences. Cluster-robust SEs (clustered by dyad).
      2) Alternative logit that adds focal group fixed effects (C(focal)) to control for time-invariant group-level heterogeneity. Cluster-robust SEs by dyad.

    Returns a dictionary with fitted model results (clustered) and prints summaries.
    """
    # Ensure required standardized columns exist
    required_z = ['z_size_diff', 'z_rel_distance', 'z_m_diff', 'z_f_diff', 'win', 'dyad', 'focal']
    missing = [c for c in required_z if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required transformed columns for modeling: {missing}")

    # Primary model: use logit
    formula_primary = 'win ~ z_size_diff * z_rel_distance + z_m_diff + z_f_diff'
    primary_logit = smf.logit(formula=formula_primary, data=df).fit(disp=False)

    # Clustered (dyad) robust SEs for inference
    clustered_primary = _make_clustered_results(primary_logit, df['dyad'], name='primary_logit')

    # Alternative model with focal fixed effects (categorical focal) to account for group-level heterogeneity
    formula_fe = 'win ~ z_size_diff * z_rel_distance + z_m_diff + z_f_diff + C(focal)'
    fe_logit = smf.logit(formula=formula_fe, data=df).fit(disp=False)
    clustered_fe = _make_clustered_results(fe_logit, df['dyad'], name='fe_logit')

    # Print model summaries (original fit summary and clustered summaries for inference)
    print('\nPrimary model (MLE fit summary)')
    print(primary_logit.summary())

    print('\nPrimary model (clustered by dyad)')
    print(clustered_primary.summary())

    print('\nFixed-effects model with C(focal) (MLE fit summary)')
    print(fe_logit.summary())

    print('\nFixed-effects model (clustered by dyad)')
    print(clustered_fe.summary())

    # Return the fitted model objects and clustered-covariance-adjusted results for programmatic use
    results = {
        'primary_logit': primary_logit,
        'primary_logit_clustered': clustered_primary,
        'fe_logit': fe_logit,
        'fe_logit_clustered': clustered_fe
    }
    return results