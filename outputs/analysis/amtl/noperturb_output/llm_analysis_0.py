from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
import scipy

from statsmodels.stats.sandwich_covariance import cov_cluster


df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/noperturb_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Ensure essential columns exist and coerce types
    # Drop rows with missing critical data needed for the binomial model
    required = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen']
    df = df.dropna(subset=required)

    # Ensure numeric columns are numeric
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')

    # Drop any rows that became NA after coercion
    df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male'])

    # Keep only rows with at least one observable socket
    df = df[df['sockets'] > 0].copy()

    # Cap num_amtl to sockets in case of coding errors
    df['num_amtl'] = df['num_amtl'].clip(lower=0)
    # Ensure num_amtl does not exceed sockets
    df['num_amtl'] = df[['num_amtl', 'sockets']].min(axis=1)

    # Proportion missing for binomial modeling
    df['prop_miss'] = df['num_amtl'] / df['sockets']

    # Primary independent variable: human vs non-human
    # Normalize genus strings then create indicator
    df['genus'] = df['genus'].astype(str).str.strip()
    df['is_human'] = (df['genus'].str.lower() == 'homo sapiens').astype(int)

    # Center age to improve interpretability and numerical stability; include quadratic term for nonlinearity
    df['age_c'] = df['age'] - df['age'].mean()
    df['age_c2'] = df['age_c'] ** 2

    # Create explicit dummy variables for tooth class, using 'Anterior' as the reference category
    df['tooth_class_Posterior'] = (df['tooth_class'].astype(str) == 'Posterior').astype(int)
    df['tooth_class_Premolar'] = (df['tooth_class'].astype(str) == 'Premolar').astype(int)

    # Keep only columns needed for modeling (plus any useful metadata)
    keep_cols = [
        'specimen',
        'num_amtl',
        'sockets',
        'prop_miss',
        'is_human',
        'age_c',
        'age_c2',
        'prob_male',
        'tooth_class_Posterior',
        'tooth_class_Premolar',
        'genus',
        'tooth_class'
    ]
    df = df.loc[:, [c for c in keep_cols if c in df.columns]]

    # Final drop of any rows with NA in the kept columns
    df = df.dropna(subset=['prop_miss', 'is_human', 'age_c', 'prob_male', 'specimen'])

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial (logistic) GLM for number of missing teeth out of sockets,
    using proportion response with binomial denominator = sockets.

    Returns a dict with the fitted GLMResults, a clustered-robust-results-like
    object (clustered by specimen), and some diagnostics (dispersion).
    """
    # Build formula: proportion as response, model with var_weights = sockets
    formula = 'prop_miss ~ is_human + age_c + age_c2 + prob_male + tooth_class_Posterior + tooth_class_Premolar'

    # Fit GLM with binomial family; use var_weights=sockets so that the model
    # treats prop_miss as num_amtl/sockets with sockets trials
    model_glm = sm.GLM.from_formula(formula, data=df, family=sm.families.Binomial(), var_weights=df['sockets'])
    results = model_glm.fit()

    # Compute clustered (by specimen) robust covariance (sandwich) to account for
    # within-specimen correlation (specimen may appear multiple times across tooth classes)
    clustered = None
    if 'specimen' in df.columns:
        try:
            cov = cov_cluster(results, df['specimen'])
            # Ensure cov is a DataFrame with appropriate index and columns
            params = results.params
            cov_df = pd.DataFrame(cov, index=params.index, columns=params.index)

            class ClusteredResults:
                def __init__(self, params: pd.Series, cov_df: pd.DataFrame):
                    self.params = params
                    self._cov = cov_df

                def cov_params(self):
                    return self._cov

                @property
                def bse(self):
                    return np.sqrt(np.diag(self._cov))

                def conf_int(self, alpha=0.05):
                    z = scipy.stats.norm.ppf(1 - alpha / 2)
                    se = self.bse
                    lower = self.params - z * se
                    upper = self.params + z * se
                    return pd.DataFrame({0: lower, 1: upper}, index=self.params.index)

            clustered = ClusteredResults(params=results.params, cov_df=cov_df)
        except Exception:
            # Fallback: use the model results' covariance matrix (non-clustered)
            clustered = results

    else:
        clustered = results

    # Compute a simple overdispersion metric: Pearson chi2 / df_resid
    # For binomial GLM fitted with weights, Pearson chi2 can still indicate overdispersion
    res = results
    pearson_chi2 = sum(res.resid_pearson ** 2)
    df_resid = res.df_resid
    dispersion = pearson_chi2 / df_resid if df_resid > 0 else np.nan

    # Prepare odds ratios and CIs using clustered covariance if available
    # clustered may be our ClusteredResults or a statsmodels results object
    if hasattr(clustered, 'params'):
        params = clustered.params
    else:
        params = results.params

    try:
        conf = clustered.conf_int()
    except Exception:
        conf = results.conf_int()

    or_vals = np.exp(params)
    # conf is expected to have columns 0 and 1
    or_ci_lower = np.exp(conf[0])
    or_ci_upper = np.exp(conf[1])

    # Pack results to return
    out = {
        'glm_results': results,
        'clustered_results': clustered,
        'dispersion': dispersion,
        'odds_ratios': or_vals,
        'or_conf_int_lower': or_ci_lower,
        'or_conf_int_upper': or_ci_upper,
        'formula': formula
    }

    return out