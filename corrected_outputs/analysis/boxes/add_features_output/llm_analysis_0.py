from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster
from scipy import stats

# If running as a script, the following path read can be done externally.
# df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/boxes/add_features_output/boxes.csv')


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe suitable for modeling the emergence of majority preference.

    Produces the following columns used in modeling:
      - MajorityChoice: binary outcome (1 if y==2 [majority], 0 otherwise)
      - Age_c: age centered at the sample mean
      - Age2: quadratic term (Age_c squared)
      - culture: cast to categorical (kept as original column name but converted to category dtype)
      - IsBoy: 1 if gender == 2 (boy), 0 if gender == 1 (girl)
      - MajorityFirst: numeric copy of majority_first (if present) otherwise NA
      - religiousness: numeric (coerced) or NA if missing
      - school: kept as provided (or created as NA if missing)

    Rows missing the essential variables (y, age, culture, IsBoy) are dropped.
    """
    df = df.copy()

    # Drop rows missing core variables required to derive key columns
    df = df.dropna(subset=['y', 'age', 'culture'])

    # Outcome: majority choice (y==2 indicates majority option)
    df['MajorityChoice'] = (df['y'] == 2).astype(int)

    # Age: ensure numeric, center and create quadratic term to capture non-linear development
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['Age_c'] = df['age'] - df['age'].mean()
    df['Age2'] = df['Age_c'] ** 2

    # Culture: ensure categorical dtype for formulas and clustering
    df['culture'] = df['culture'].astype('category')

    # Gender -> binary control (1 = boy, 0 = girl). If gender missing, set NA.
    if 'gender' in df.columns:
        gender_num = pd.to_numeric(df['gender'], errors='coerce')
        # Use float dtype with np.nan for missing so patsy/statsmodels handle it robustly.
        df['IsBoy'] = np.where(gender_num.isna(), np.nan, (gender_num == 2).astype(float))
    else:
        df['IsBoy'] = np.nan

    # Majority demonstration order as numeric control (if present). Keep NA if missing.
    if 'majority_first' in df.columns:
        df['MajorityFirst'] = pd.to_numeric(df['majority_first'], errors='coerce').astype(float)
    else:
        df['MajorityFirst'] = np.nan

    # Ensure religiousness is numeric (coerce and keep NA if invalid or missing)
    if 'religiousness' in df.columns:
        df['religiousness'] = pd.to_numeric(df['religiousness'], errors='coerce').astype(float)
    else:
        df['religiousness'] = np.nan

    # Ensure school column exists (kept for clustering / multilevel analyses). If missing, create NA column.
    if 'school' not in df.columns:
        df['school'] = np.nan

    # Drop rows missing truly essential final variables that must be present for modeling:
    # - MajorityChoice (outcome)
    # - Age_c (predictor)
    # - culture (moderator / cluster)
    # - IsBoy (control)
    #
    # We intentionally do NOT drop rows solely for missing MajorityFirst or religiousness here;
    # statsmodels will handle row-wise NA filtering during model fitting.
    df = df.dropna(subset=['MajorityChoice', 'Age_c', 'culture', 'IsBoy'])

    # Ensure final columns exist with the exact required names (even if they contain NA)
    for col in ['MajorityChoice', 'Age_c', 'Age2', 'culture', 'IsBoy', 'MajorityFirst', 'religiousness', 'school']:
        if col not in df.columns:
            df[col] = np.nan

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression predicting the probability of choosing the majority option.

    Primary model: a GLM with binomial family where the key terms are Age_c and its interaction with culture
    to test whether developmental trajectories (age slopes) differ across cultural contexts.

    Formula used:
      MajorityChoice ~ Age_c * C(culture) + Age2 + IsBoy + MajorityFirst + religiousness

    We compute cluster-robust standard errors clustered by culture (the moderator) to account for within-site dependence.

    Returns the fitted results object with additional attributes containing clustered covariance and related statistics.
    """
    df = df.copy()

    if 'culture' not in df.columns:
        raise ValueError("'culture' column required in dataframe")

    formula = 'MajorityChoice ~ Age_c * C(culture) + Age2 + IsBoy + MajorityFirst + religiousness'

    # Fit GLM (binomial)
    glm_model = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
    res = glm_model.fit()

    # Prepare cluster grouping vector aligned to the rows used by the fitted model
    # Use categorical codes if available; keep as a pandas Series with the same index as df for alignment.
    if hasattr(df['culture'], 'cat'):
        groups_series = df['culture'].cat.codes
    else:
        groups_series = pd.Series(df['culture'].values, index=df.index)

    # Align group labels to the observations actually used in the fit (model data)
    try:
        model_data = res.model.data
        design_index = None
        # Prefer the frame's index if available
        if hasattr(model_data, 'frame') and getattr(model_data, 'frame') is not None:
            try:
                design_index = model_data.frame.index
            except Exception:
                design_index = None
        # Fallback to row_labels if frame not available
        if design_index is None and hasattr(model_data, 'row_labels'):
            try:
                design_index = pd.Index(model_data.row_labels)
            except Exception:
                design_index = None

        if design_index is not None:
            groups_aligned = groups_series.reindex(design_index)
            groups_array = groups_aligned.values
        else:
            groups_array = groups_series.values
    except Exception:
        # Fallback: use the groups in index order of the provided dataframe
        groups_array = groups_series.values

    # Compute clustered covariance matrix
    cluster_vcov = cov_cluster(res, groups_array)

    # Attach cluster-robust metrics to the results object for downstream use
    bse_cluster = np.sqrt(np.diag(cluster_vcov))
    tvalues_cluster = res.params / bse_cluster
    pvalues_cluster = 2 * stats.norm.sf(np.abs(tvalues_cluster))

    # Attach attributes
    setattr(res, 'cov_cluster', cluster_vcov)
    setattr(res, 'bse_cluster', bse_cluster)
    setattr(res, 'tvalues_cluster', tvalues_cluster)
    setattr(res, 'pvalues_cluster', pvalues_cluster)

    # Print concise table similar to a summary for cluster-robust results
    try:
        summary_table = pd.DataFrame({
            'coef': res.params,
            'bse_cluster': bse_cluster,
            'z_cluster': tvalues_cluster,
            'p_cluster': pvalues_cluster
        })
        print(summary_table)
    except Exception:
        # If anything goes wrong printing, still return the results with attached attributes
        pass

    return res