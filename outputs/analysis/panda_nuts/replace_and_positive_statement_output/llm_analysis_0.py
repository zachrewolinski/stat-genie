from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster
from scipy import stats as _stats


df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/panda_nuts/replace_and_positive_statement_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the chimpanzee nut-cracking dataset for modeling.

    Output columns required by the model:
      - nuts_opened (int)        : dependent count
      - seconds (float)          : exposure (duration of session in seconds)
      - rate_per_min (float)     : nuts opened per minute (derived descriptive DV)
      - log_seconds (float)      : natural log of seconds (offset for count models)
      - age (float)              : original age (kept)
      - age_c (float)            : age centered (predictor)
      - sex_M (int)              : sex indicator (1 = male, 0 = female)
      - help_Y (int)             : help indicator (1 = yes, 0 = no)
      - hammer (category)        : hammer type (control, kept as categorical)
      - chimpanzee (int)         : individual id (for clustering)

    Rows with missing key fields or nonpositive seconds are dropped.
    """

    df = df.copy()

    # Keep only relevant columns if dataset contains extras
    needed = ['chimpanzee', 'age', 'sex', 'hammer', 'nuts_opened', 'seconds', 'help']
    for col in needed:
        if col not in df.columns:
            raise KeyError(f"Required column missing from input dataframe: {col}")

    # Drop rows with missing values in key columns
    df = df.dropna(subset=['chimpanzee', 'age', 'sex', 'nuts_opened', 'seconds', 'help'])

    # Ensure numeric types
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Drop rows with invalid numeric values
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age'])
    # Remove sessions with non-positive seconds (can't be used as exposure)
    df = df[df['seconds'] > 0]

    # Standardize / encode predictors
    # sex: create male indicator (sex_M = 1 if male)
    # Accept common male labels starting with 'm' (case-insensitive)
    df['sex_M'] = df['sex'].astype(str).str.strip().str.lower().map(lambda x: 1 if x.startswith('m') else 0)

    # help: create indicator (help_Y = 1 if helped)
    df['help_Y'] = df['help'].astype(str).str.strip().str.lower().map(lambda x: 1 if x in ['y', 'yes', '1', 'true', 't'] else 0)

    # Derived descriptive rate: nuts per minute (useful for plotting and diagnostics)
    df['rate_per_min'] = df['nuts_opened'] / (df['seconds'] / 60.0)

    # Exposure for count models: seconds and its log
    df['log_seconds'] = np.log(df['seconds'].astype(float))

    # Keep hammer as categorical (modeling will use C(hammer) in formula)
    df['hammer'] = df['hammer'].astype('category')

    # Ensure chimpanzee id is integer (or category) for clustering
    try:
        df['chimpanzee'] = df['chimpanzee'].astype(int)
    except Exception:
        # fallback to categorical codes if not integer-parseable
        df['chimpanzee'] = pd.Categorical(df['chimpanzee']).codes

    # Center age for numerical stability and interpretability
    df['age_c'] = df['age'] - df['age'].mean()

    # Final check — ensure required columns exist and have no missing
    final_cols = ['nuts_opened', 'seconds', 'rate_per_min', 'log_seconds', 'age', 'age_c', 'sex_M', 'help_Y', 'hammer', 'chimpanzee']
    missing_final = [c for c in final_cols if c not in df.columns]
    if missing_final:
        raise RuntimeError(f"Transform failed to produce required columns: {missing_final}")

    # Ensure no missing values in final required columns
    df = df.dropna(subset=final_cols)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a count model for nuts opened using session duration as exposure.

    Strategy:
      1. Fit a Poisson GLM with offset = log(seconds).
      2. Compute dispersion (Pearson chi2 / df_resid). If dispersion > 1.5, refit with Negative Binomial family.
      3. Return the final fitted model object and a small summary table with incidence-rate ratios (IRRs) and clustered SEs by chimpanzee.

    Model formula:
      nuts_opened ~ age_c + sex_M + help_Y + C(hammer)

    The returned object is a simple wrapper providing clustered results info. The function also prints summary and IRRs.
    """

    # Ensure the dataframe has the columns we expect
    required = ['nuts_opened', 'seconds', 'log_seconds', 'age_c', 'sex_M', 'help_Y', 'hammer', 'chimpanzee']
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Required column for modeling missing: {c}")

    # Define formula
    formula = 'nuts_opened ~ age_c + sex_M + help_Y + C(hammer)'

    # Fit Poisson GLM with offset = log_seconds
    poisson_model = smf.glm(formula=formula, data=df, family=sm.families.Poisson(), offset=df['log_seconds'])
    poisson_res = poisson_model.fit()

    # Compute dispersion: Pearson chi2 / df_resid
    try:
        pearson_chi2 = np.sum(poisson_res.resid_pearson ** 2)
    except Exception:
        mu = poisson_res.fittedvalues
        y = poisson_res.model.endog
        pearson_chi2 = np.sum((y - mu) ** 2 / np.where(mu == 0, 1e-8, mu))

    dispersion = pearson_chi2 / poisson_res.df_resid if poisson_res.df_resid > 0 else np.nan

    # Choose family based on dispersion
    if not np.isnan(dispersion) and dispersion > 1.5:
        chosen_family = sm.families.NegativeBinomial()
        fam_name = 'NegativeBinomial'
    else:
        chosen_family = sm.families.Poisson()
        fam_name = 'Poisson'

    # Fit final model with chosen family
    final_model = smf.glm(formula=formula, data=df, family=chosen_family, offset=df['log_seconds'])
    final_res = final_model.fit()

    # Compute clustered (by chimpanzee) robust covariance matrix
    # Use statsmodels' cov_cluster to get clustered covariance, then build a small wrapper
    group = df['chimpanzee'].values
    clustered_cov = cov_cluster(final_res, group)

    params = final_res.params.copy()
    # Ensure params is a pandas Series with proper index
    if not isinstance(params, pd.Series):
        params = pd.Series(params, index=final_res.params.index)

    # standard errors from clustered covariance
    bse_cluster = np.sqrt(np.diag(clustered_cov))
    bse_cluster = pd.Series(bse_cluster, index=params.index)

    # z-statistics and p-values
    z_stats = params / bse_cluster
    pvalues = 2 * (1 - _stats.norm.cdf(np.abs(z_stats)))
    pvalues = pd.Series(pvalues, index=params.index)

    # 95% CI based on normal approximation
    crit = _stats.norm.ppf(0.975)
    conf_lower = params - crit * bse_cluster
    conf_upper = params + crit * bse_cluster
    conf_df = pd.DataFrame({0: conf_lower, 1: conf_upper})

    # Simple wrapper object to mimic minimal interface expected by downstream code
    class ClusteredResults:
        def __init__(self, params, bse, pvalues, conf_df, final_res, clustered_cov):
            self.params = params
            self.bse = bse
            self.pvalues = pvalues
            self._conf = conf_df
            self._final = final_res
            self.cov = clustered_cov

        def conf_int(self):
            return self._conf

        def summary(self):
            # Provide the original (model-based) summary plus a note about clustered SEs
            base = ""
            try:
                base = self._final.summary().as_text()
            except Exception:
                base = str(self._final.summary())
            note = "\nNote: standard errors, p-values, and CIs below are computed with clustering by 'chimpanzee'."
            return base + note

        def __repr__(self):
            return self.summary()

    clustered_res = ClusteredResults(params=params, bse=bse_cluster, pvalues=pvalues, conf_df=conf_df, final_res=final_res, clustered_cov=clustered_cov)

    # Prepare IRR table (exp(coef)) with clustered CIs
    irr = np.exp(clustered_res.params)
    irr_conf = np.exp(clustered_res.conf_int())
    irr_table = pd.DataFrame({
        'coef': clustered_res.params,
        'IRR': irr,
        'IRR_CI_lower': irr_conf[0],
        'IRR_CI_upper': irr_conf[1],
        'pvalue': clustered_res.pvalues
    })

    # Print model selection diagnostics and summary
    print("Model family chosen:", fam_name)
    print(f"Dispersion (Pearson chi2 / df_resid) from initial Poisson: {dispersion:.3f}")
    print('\nClustered robust summary:')
    print(clustered_res.summary())
    print('\nIncidence-rate ratios (IRR) with clustered CIs:')
    print(irr_table)

    # Return a dict with useful objects
    return {
        'final_results_clustered': clustered_res,
        'irr_table': irr_table,
        'dispersion': dispersion,
        'family': fam_name,
        'poisson_initial_results': poisson_res
    }