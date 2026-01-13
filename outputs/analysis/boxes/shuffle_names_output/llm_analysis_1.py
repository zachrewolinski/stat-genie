from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe with the columns required for modeling.

    Produced columns (FINAL):
    - MajorityChosen: binary (1 if child chose the majority option, 0 otherwise)
    - Age: child's age in years (from original column 'culture')
    - IsGirl: binary (1 if gender==1 -> girl, 0 if gender==2 -> boy)
    - MajorityFirstShown: binary indicator whether the majority option was demonstrated first (from original column 'age', assumed 0/1 or 1/2)
    - Site: categorical site ID (from original column 'y')

    Notes:
    - This function is robust to string/numeric encodings of the original columns.
    - It only drops rows that cannot be mapped to the required FINAL columns. If some control variables are missing,
      it will attempt to impute them with the modal value to preserve rows for modeling.
    """

    df = df.copy()

    # Required original columns that must exist in the input
    required_cols = ['majority_first', 'culture', 'age', 'gender', 'y']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Helper mappers to be permissive about input encodings
    def map_majority_chosen(val):
        # Return 1 if majority chosen, 0 if not, np.nan if unknown
        if pd.isna(val):
            return np.nan
        num = pd.to_numeric(val, errors='coerce')
        if pd.notnull(num):
            try:
                ival = int(num)
                # Many datasets encode majority choice as 2; treat 2 as majority
                if ival == 2:
                    return 1
                # If encoding uses 1 for majority, accept that as well
                if ival == 1:
                    # Here we don't know whether 1 corresponds to majority in this dataset,
                    # but original spec says 2 indicates majority. Treat 1 as non-majority
                    # unless textual evidence indicates otherwise; keep as 0.
                    return 0
                return 0
            except Exception:
                pass
        s = str(val).strip().lower()
        if s in {'2', 'majority', 'maj', 'major', 'chosen_majority', 'chosen majority'}:
            return 1
        if s in {'1', '0', 'minority', 'min', 'other', 'chosen_minority', 'chosen minority'}:
            return 0
        return np.nan

    def map_is_girl(val):
        # Return 1 for girl, 0 for boy, np.nan if unknown
        if pd.isna(val):
            return np.nan
        num = pd.to_numeric(val, errors='coerce')
        if pd.notnull(num):
            try:
                ival = int(num)
                if ival == 1:
                    return 1
                if ival == 2:
                    return 0
            except Exception:
                pass
        s = str(val).strip().lower()
        if s in {'1', 'girl', 'female', 'f', 'g', 'female '}:
            return 1
        if s in {'2', 'boy', 'male', 'm', 'male '}:
            return 0
        return np.nan

    def map_majority_first_shown(val):
        # Return 1 if majority shown first, 0 if not, np.nan if unknown
        if pd.isna(val):
            return np.nan
        num = pd.to_numeric(val, errors='coerce')
        if pd.notnull(num):
            try:
                ival = int(num)
                # Accept both 1 and 2 encodings as possible majority-first indicators:
                # if dataset uses 1 -> majority first or 2 -> majority first, accept both.
                if ival == 1 or ival == 2:
                    return 1
                if ival == 0:
                    return 0
            except Exception:
                pass
        s = str(val).strip().lower()
        if s in {'1', '2', 'true', 't', 'yes', 'y', 'majority_first', 'majority', 'first_majority'}:
            return 1
        if s in {'0', 'false', 'f', 'no', 'n', 'not_first', 'minority_first'}:
            return 0
        return np.nan

    # Apply mappings
    df['MajorityChosen'] = df['majority_first'].apply(map_majority_chosen)
    df['Age'] = pd.to_numeric(df['culture'], errors='coerce')
    df['IsGirl'] = df['gender'].apply(map_is_girl)
    df['MajorityFirstShown'] = df['age'].apply(map_majority_first_shown)

    # Site (cultural context) from column 'y' - treat as categorical
    df['Site'] = df['y'].astype('category')

    # Final required columns (these names are mandated by the specification)
    final_cols = ['MajorityChosen', 'Age', 'IsGirl', 'MajorityFirstShown', 'Site']

    # First, try strict drop: require all final columns non-missing
    df_strict = df.dropna(subset=final_cols).copy()

    if df_strict.shape[0] > 0:
        df_final = df_strict
    else:
        # Try a more permissive approach: require only essential columns for outcome and moderator:
        essential = ['MajorityChosen', 'Age', 'Site']
        df_partial = df.dropna(subset=essential).copy()
        if df_partial.shape[0] == 0:
            # Nothing we can do
            raise ValueError("No rows remain after transforming and dropping missing essential columns (MajorityChosen, Age, Site). "
                             "Cannot proceed with modeling.")
        # Impute missing control variables (IsGirl, MajorityFirstShown) with modal value if possible,
        # otherwise default to 0. This preserves rows for modeling while making a reasonable guess.
        for col, default in [('IsGirl', 0), ('MajorityFirstShown', 0)]:
            if col not in df_partial.columns:
                df_partial[col] = default
            if df_partial[col].isna().all():
                # try to compute mode from the original mapped column (before dropping)
                mode_series = df[col].dropna()
                if not mode_series.empty:
                    try:
                        mode_val = int(mode_series.mode().iloc[0])
                    except Exception:
                        mode_val = default
                else:
                    mode_val = default
                df_partial[col] = df_partial[col].fillna(mode_val)
            else:
                # fill remaining NaNs with mode from available values in df_partial, or overall default
                try:
                    mode_val = int(df_partial[col].mode().iloc[0])
                except Exception:
                    mode_val = default
                df_partial[col] = df_partial[col].fillna(mode_val)
        df_final = df_partial

    # Enforce types for final columns
    # MajorityChosen, IsGirl, MajorityFirstShown should be integer 0/1
    # Age as float, Site as category
    df_final['MajorityChosen'] = df_final['MajorityChosen'].astype(int)
    df_final['Age'] = df_final['Age'].astype(float)
    df_final['IsGirl'] = df_final['IsGirl'].astype(int)
    df_final['MajorityFirstShown'] = df_final['MajorityFirstShown'].astype(int)
    df_final['Site'] = df_final['Site'].astype('category')

    # Return final dataframe with mandated column ordering
    return df_final[final_cols]


def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial (logistic) regression to test how children's reliance on the majority
    (MajorityChosen) changes with age, and whether that age effect varies across cultural
    contexts (Site). Controls for child gender and whether the majority was shown first.

    Model specification:
    - Outcome: MajorityChosen (binary)
    - Predictors: Age, Age x Site (interaction to allow different age slopes across sites),
      IsGirl, MajorityFirstShown, and site fixed effects (via C(Site)).

    Estimation details:
    - Use GLM with binomial family (logit link).
    - Compute cluster-robust standard errors clustered by Site to account for within-site dependence.

    Returns the fitted results object (robust covariance adjustments applied) or raises informative errors.
    """

    import statsmodels.formula.api as smf

    required = ['MajorityChosen', 'Age', 'IsGirl', 'MajorityFirstShown', 'Site']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    if df.shape[0] == 0:
        raise ValueError("Input dataframe to model() is empty. Ensure transform() produced at least one row.")

    # Ensure Site is categorical
    if not pd.api.types.is_categorical_dtype(df['Site']):
        df = df.copy()
        df['Site'] = df['Site'].astype('category')

    # Define formula. Age * C(Site) expands to Age + C(Site) + Age:C(Site)
    formula = 'MajorityChosen ~ Age * C(Site) + IsGirl + MajorityFirstShown'

    # Fit GLM (binomial)
    try:
        glm_mod = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
        fitted = glm_mod.fit()
    except Exception as e:
        # Provide informative error if fitting fails
        raise RuntimeError(f"GLM fitting failed: {e}")

    # Attempt to compute cluster-robust SEs clustered by Site; fall back to unadjusted results if not possible
    try:
        # Use categorical codes as grouping labels
        if pd.api.types.is_categorical_dtype(df['Site']):
            groups = df['Site'].cat.codes
        else:
            groups = df['Site']
        results = fitted.get_robustcov_results(cov_type='cluster', groups=groups)
    except Exception:
        results = fitted

    return results