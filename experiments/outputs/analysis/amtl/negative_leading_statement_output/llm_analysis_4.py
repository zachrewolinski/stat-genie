from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
import patsy

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/negative_leading_statement_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw Gilmore (2013) AMTL dataset into an analysis-ready dataframe.

    Produced columns used in the model:
      - amtl_successes: integer count of missing teeth (num_amtl, clipped to [0, sockets])
      - amtl_trials: integer count of observable sockets (sockets)
      - is_human: binary indicator (1 = 'Homo sapiens', 0 = other genus)
      - age_center: age minus mean age (centering improves interpretability/stability)
      - prob_male: numeric sex estimate (0..1) (kept as provided)
      - tooth_class: categorical with consistent categories
      - specimen: specimen identifier (kept for clustering)

    Notes:
      - Rows with non-positive amtl_trials (sockets <= 0) are removed because
        binomial proportions are undefined for zero trials.
      - tooth_class values are normalized to the three expected categories and
        rows that cannot be mapped are removed.
    """
    df = df.copy()

    # Drop rows missing essential fields first (raw checks)
    df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen'])

    # Ensure integer counts for successes and trials
    df['amtl_successes'] = df['num_amtl'].astype(int)
    df['amtl_trials'] = df['sockets'].astype(int)

    # Safety: clip successes to the valid range [0, trials]
    df['amtl_successes'] = df[['amtl_successes', 'amtl_trials']].apply(
        lambda row: max(0, min(int(row['amtl_successes']), int(row['amtl_trials']))), axis=1
    )

    # Remove rows with zero or negative trials (sockets) to avoid 0/0 proportions
    df = df[df['amtl_trials'] > 0].copy()

    # Binary indicator for modern human (Homo sapiens), robust to spacing/casing
    df['is_human'] = (df['genus'].astype(str).str.strip().str.lower() == 'homo sapiens').astype(int)

    # Center age (mean-centering)
    df['age_center'] = df['age'] - df['age'].mean()

    # Ensure prob_male numeric and within [0,1]
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')
    # Any rows with invalid prob_male become NaN; drop them
    df = df.dropna(subset=['prob_male'])
    df['prob_male'] = df['prob_male'].clip(0.0, 1.0)

    # Normalize tooth_class values into the three categories: Anterior, Premolar, Posterior
    def map_to_class(x: str) -> Optional[str]:
        if x is None:
            return None
        s = str(x).strip().lower()
        if not s:
            return None
        # simple substring matching to capture common variants
        if 'ant' in s:
            return 'Anterior'
        if 'prem' in s or 'pm' in s:
            return 'Premolar'
        if 'post' in s:
            return 'Posterior'
        # fallback: exact matches
        if s in ('anterior', 'premolar', 'posterior'):
            return s.title()
        return None

    df['tooth_class_mapped'] = df['tooth_class'].apply(map_to_class)

    # Drop rows that couldn't be mapped to one of the expected tooth classes
    df = df.dropna(subset=['tooth_class_mapped']).copy()

    # Set tooth_class categorical with consistent ordering
    df['tooth_class'] = pd.Categorical(df['tooth_class_mapped'],
                                       categories=['Anterior', 'Premolar', 'Posterior'])

    # Keep only the columns required for modeling and inspection
    keep_cols = ['specimen', 'pop', 'genus', 'tooth_class',
                 'amtl_successes', 'amtl_trials', 'age', 'age_center', 'prob_male', 'is_human']
    # If 'pop' is not present for some reason, keep operation still safe
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # Final defensive checks: ensure required conceptual columns exist
    required_cols = ['is_human', 'amtl_successes', 'amtl_trials', 'age_center', 'prob_male', 'tooth_class', 'specimen']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Transformed dataframe is missing required columns: {missing}")

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a binomial GLM to test whether modern humans (is_human == 1) have higher AMTL rates
    than non-human primates after controlling for age, sex, and tooth class.

    Modeling approach:
      - Response: amtl_successes out of amtl_trials (binomial counts)
      - Family: Binomial
      - Predictors: is_human + age_center + prob_male + C(tooth_class)
      - Inference: report both model fit and cluster-robust SEs (clustered by specimen)
      - Compute exponentiated coefficient (odds ratio) and 95% CI for is_human.
      - Compute dispersion (deviance / df_resid) as a check for overdispersion.

    Returns a dictionary containing the fitted models and key summaries.
    """
    import statsmodels.api as sm
    from patsy import dmatrix, build_design_matrices

    # Work on a copy
    df = df.copy()

    # Defensive checks: ensure no zero trials and no NaNs in key columns
    df = df.dropna(subset=['amtl_successes', 'amtl_trials', 'is_human', 'age_center', 'prob_male', 'tooth_class', 'specimen'])
    df = df[df['amtl_trials'] > 0].copy()

    # Prepare endog as two-column array: [successes, failures]
    successes = np.asarray(df['amtl_successes'], dtype=float)
    failures = np.asarray(df['amtl_trials'] - df['amtl_successes'], dtype=float)
    # Ensure non-negative failures
    failures = np.where(failures < 0, 0.0, failures)
    endog = np.column_stack((successes, failures))

    # Build design matrix for predictors using patsy to handle categorical encoding
    exog = dmatrix('is_human + age_center + prob_male + C(tooth_class)', df, return_type='dataframe')

    # Fit GLM with Binomial family using counts (endog as (n,2))
    glm_model = sm.GLM(endog, exog, family=sm.families.Binomial())

    res = glm_model.fit()

    # Check for overdispersion: deviance / residual df
    dispersion = float(res.deviance / res.df_resid) if res.df_resid > 0 else np.nan

    # Obtain cluster-robust covariance for inference (clustered by specimen)
    try:
        # Use get_robustcov_results to attach clustered cov
        res_cluster = res.get_robustcov_results(cov_type='cluster', groups=df['specimen'])
    except Exception:
        # Fallback: use default (non-clustered) results when clustering fails
        res_cluster = res

    # Extract coefficient, odds ratio, and 95% CI for is_human
    # The parameter name should be exactly 'is_human' because we used that variable in the design matrix
    param_name = 'is_human'
    if param_name in res_cluster.params.index:
        coef = float(res_cluster.params[param_name])
        or_val = float(np.exp(coef))
        ci = res_cluster.conf_int().loc[param_name].astype(float)
        ci_or = list(np.exp(ci.values))
    else:
        # In some encoding schemes, the name might differ; try to find a matching parameter
        matches = [p for p in res_cluster.params.index if p.endswith('is_human') or p == 'is_human']
        if matches:
            p0 = matches[0]
            coef = float(res_cluster.params[p0])
            or_val = float(np.exp(coef))
            ci = res_cluster.conf_int().loc[p0].astype(float)
            ci_or = list(np.exp(ci.values))
        else:
            coef = np.nan
            or_val = np.nan
            ci_or = [np.nan, np.nan]

    # Compute predicted probabilities for a representative case (average covariates)
    mean_age_center = df['age_center'].mean() if 'age_center' in df.columns else 0.0
    mean_prob_male = df['prob_male'].mean() if 'prob_male' in df.columns else 0.0

    # choose a tooth_class that exists in the data for prediction (fallback to first category)
    if hasattr(df['tooth_class'], 'cat'):
        tooth_cats = [c for c in df['tooth_class'].cat.categories if c in df['tooth_class'].unique()]
        if len(tooth_cats) == 0:
            tooth_choice = df['tooth_class'].cat.categories[0]
        else:
            tooth_choice = tooth_cats[-1]
    else:
        unique_tc = sorted(df['tooth_class'].unique())
        tooth_choice = unique_tc[-1] if unique_tc else None

    if tooth_choice is not None:
        pred_rows = [
            {'is_human': 0, 'age_center': mean_age_center, 'prob_male': mean_prob_male, 'tooth_class': tooth_choice},
            {'is_human': 1, 'age_center': mean_age_center, 'prob_male': mean_prob_male, 'tooth_class': tooth_choice}
        ]
        pred_df = pd.DataFrame(pred_rows)
        # Ensure tooth_class has same categories as training data (if categorical)
        if hasattr(df['tooth_class'], 'cat'):
            pred_df['tooth_class'] = pd.Categorical(pred_df['tooth_class'], categories=df['tooth_class'].cat.categories)
        # Build design matrix for prediction using original design info
        try:
            design_info = exog.design_info
            exog_pred = patsy.build_design_matrices([design_info], pred_df, return_type='dataframe')[0]
        except Exception:
            # As a fallback, construct design via same formula (should work in most cases)
            exog_pred = dmatrix('is_human + age_center + prob_male + C(tooth_class)', pred_df, return_type='dataframe')

        pred_probs = res_cluster.predict(exog_pred)
        # Ensure we handle both numpy arrays and pandas objects
        pred_prob_nonhuman = float(pred_probs[0])
        pred_prob_human = float(pred_probs[1])
    else:
        pred_prob_nonhuman = np.nan
        pred_prob_human = np.nan

    results = {
        'glm_result': res,
        'glm_result_cluster': res_cluster,
        'dispersion': dispersion,
        'coef_is_human': coef,
        'odds_ratio_is_human': or_val,
        'ci_or_is_human': ci_or,
        'pred_prob_nonhuman_at_mean': pred_prob_nonhuman,
        'pred_prob_human_at_mean': pred_prob_human
    }

    return results