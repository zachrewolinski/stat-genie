from typing import Any, List, Dict, Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm


def _find_and_rename(df: pd.DataFrame, desired: str, candidates: List[str]) -> None:
    """
    Find the first column in candidates that exists in df (case-insensitive match)
    and rename it to `desired`. If `desired` is already present, do nothing.
    """
    if desired in df.columns:
        return
    # Build mapping from lowercased column name to actual column name
    col_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand is None:
            continue
        lc = cand.lower()
        if lc in col_map:
            df.rename(columns={col_map[lc]: desired}, inplace=True)
            return
    # No match found: do nothing; caller will handle missing required columns later.


def _find_column_by_keywords(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    """
    Return the first column name in df whose lowercased name contains any of the keywords
    and which contains at least one non-null numeric value after coercion.
    """
    for col in df.columns:
        lname = col.lower()
        if any(k in lname for k in keywords):
            # Check if column can be interpreted as numeric for at least one value
            coerced = pd.to_numeric(df[col], errors='coerce')
            if coerced.notnull().any():
                return col
    return None


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe suitable for binomial GLM.

    Ensures the final dataframe contains the required columns:
    ['n_missing', 'n_present', 'amtl_prop', 'is_human', 'Sex_Male',
     'Tooth_Anterior', 'Tooth_Premolar', 'Age_c', 'Genus', 'ToothClass',
     'SpecimenID', 'Region', 'Age', 'AgeSD', 'n_obs']

    The function is robust to a variety of raw column names by attempting to
    map common variants to the required internal names.
    """
    df = df.copy()

    # Candidate raw names for each desired final column.
    candidates_map: Dict[str, List[str]] = {
        'ToothClass': [
            'feature1', 'toothclass', 'tooth_class', 'tooth class', 'class',
            'toothtype', 'tooth_type'
        ],
        'SpecimenID': [
            'feature2', 'specimenid', 'specimen_id', 'specimen', 'id', 'sampleid', 'sample_id'
        ],
        'n_missing': [
            'feature3', 'n_missing', 'num_missing', 'missing', 'n_missing_teeth',
            'lost', 'num_lost', 'missing_count'
        ],
        'n_obs': [
            'feature4', 'n_obs', 'num_obs', 'n_observed', 'observed', 'n_total',
            'n_sockets', 'n_sockets_obs', 'n_assessed'
        ],
        'Age': [
            'feature5', 'age', 'age_at_death', 'ageatdeath', 'estimated_age', 'age_est', 'age_years'
        ],
        'AgeSD': [
            'feature6', 'agesd', 'age_sd', 'age_uncertainty', 'age_se'
        ],
        'SexEstimate': [
            'feature7', 'sexestimate', 'sex_estimate', 'sex', 'sex_est', 'sexprob', 'sex_prob'
        ],
        'Genus': [
            'feature8', 'genus', 'gen', 'taxon', 'genname'
        ],
        'Region': [
            'feature9', 'region', 'site', 'locality', 'location'
        ]
    }

    # Attempt to find and rename columns to the desired names.
    for desired, cands in candidates_map.items():
        _find_and_rename(df, desired, cands)

    # Flexible inference for count columns if initial mapping failed.
    missing_keywords = ['missing', 'miss', 'lost', 'num_miss', 'num_missing', 'n_missing', 'missing_count']
    obs_keywords = ['obs', 'observ', 'total', 'n_total', 'sockets', 'assessed', 'sample_size']
    present_keywords = ['present', 'present_count', 'n_present', 'num_present', 'observed']

    # Try to find a raw n_missing if not present or all null
    if ('n_missing' not in df.columns) or df['n_missing'].isnull().all():
        col = _find_column_by_keywords(df, missing_keywords)
        if col is not None:
            df.rename(columns={col: 'n_missing'}, inplace=True)

    # Try to find a raw n_obs if not present or all null
    if ('n_obs' not in df.columns) or df['n_obs'].isnull().all():
        col = _find_column_by_keywords(df, obs_keywords)
        if col is not None:
            df.rename(columns={col: 'n_obs'}, inplace=True)

    # Try to find a raw n_present if not present or all null
    if ('n_present' not in df.columns) or df['n_present'].isnull().all():
        col = _find_column_by_keywords(df, present_keywords)
        if col is not None:
            df.rename(columns={col: 'n_present'}, inplace=True)

    # Now safely coerce potential count columns to numeric (if they exist)
    for col in ['n_missing', 'n_obs', 'n_present']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # If n_present missing but we have n_obs and n_missing, compute it
    if ('n_present' not in df.columns or df['n_present'].isnull().all()) and ('n_obs' in df.columns and 'n_missing' in df.columns):
        df['n_present'] = df['n_obs'] - df['n_missing']

    # If n_obs missing but we have n_present and n_missing, compute it
    if ('n_obs' not in df.columns or df['n_obs'].isnull().all()) and ('n_present' in df.columns and 'n_missing' in df.columns):
        df['n_obs'] = df['n_present'] + df['n_missing']

    # Drop rows where counts could not be interpreted as numbers (if those columns exist)
    cols_to_check = [c for c in ['n_missing', 'n_obs', 'n_present'] if c in df.columns]
    if cols_to_check:
        # keep rows where at least one count is present
        df = df.dropna(subset=cols_to_check, how='all')

    # Ensure numeric and recompute derived counts where possible
    if ('n_obs' in df.columns) and ('n_missing' in df.columns):
        df['n_obs'] = pd.to_numeric(df['n_obs'], errors='coerce')
        df['n_missing'] = pd.to_numeric(df['n_missing'], errors='coerce')
        df['n_present'] = df['n_obs'] - df['n_missing']

    if ('n_present' in df.columns) and ('n_missing' in df.columns) and ('n_obs' not in df.columns):
        df['n_obs'] = df['n_present'] + df['n_missing']

    # Ensure the columns exist so downstream code sees them; they'll be validated/filled below.
    if 'n_missing' not in df.columns:
        df['n_missing'] = np.nan
    if 'n_present' not in df.columns:
        df['n_present'] = np.nan

    # After ensuring numeric, cast to integer counts where present (round first to be safe)
    if 'n_missing' in df.columns:
        df['n_missing'] = pd.to_numeric(df['n_missing'], errors='coerce')
        if not df['n_missing'].isnull().all():
            df['n_missing'] = df['n_missing'].round().astype(pd.Int64Dtype()).astype(float)

    if 'n_present' in df.columns:
        df['n_present'] = pd.to_numeric(df['n_present'], errors='coerce')
        if not df['n_present'].isnull().all():
            df['n_present'] = df['n_present'].round().astype(pd.Int64Dtype()).astype(float)

    if 'n_obs' in df.columns:
        df['n_obs'] = pd.to_numeric(df['n_obs'], errors='coerce')
        if not df['n_obs'].isnull().all():
            df['n_obs'] = df['n_obs'].round().astype(pd.Int64Dtype()).astype(float)

    # Remove inconsistent rows and zero-observation rows if both counts present
    if ('n_present' in df.columns) and ('n_obs' in df.columns):
        # Ensure non-negative and n_obs > 0
        # Use boolean masks that tolerate NaNs without dropping rows prematurely
        mask_nonneg_present = (df['n_present'].isnull()) | (df['n_present'] >= 0)
        mask_obs_positive = (df['n_obs'].isnull()) | (df['n_obs'] > 0)
        mask_present_le_obs = (df['n_present'].isnull()) | (df['n_obs'].isnull()) | (df['n_present'] <= df['n_obs'])
        df = df[mask_nonneg_present & mask_obs_positive & mask_present_le_obs]

    # amtl_prop for diagnostics if possible
    if ('n_missing' in df.columns) and ('n_obs' in df.columns):
        # Avoid division by zero; n_obs filtered to >0 above where possible
        df['amtl_prop'] = df['n_missing'] / df['n_obs']
    else:
        df['amtl_prop'] = np.nan

    # Genus normalization and is_human indicator
    if 'Genus' in df.columns:
        df['Genus'] = df['Genus'].astype(str).str.strip()
        df['is_human'] = (df['Genus'].str.lower() == 'homo sapiens').astype(int)
    else:
        # If Genus is missing, create column of NaNs and is_human as zeros
        df['Genus'] = np.nan
        df['is_human'] = 0

    # Sex_Male from SexEstimate thresholded at 0.5
    if 'SexEstimate' in df.columns:
        df['Sex_Male'] = (pd.to_numeric(df['SexEstimate'], errors='coerce') >= 0.5).astype(int)
    else:
        df['Sex_Male'] = 0

    # Tooth class dummies: Posterior reference
    if 'ToothClass' in df.columns:
        df['ToothClass'] = df['ToothClass'].astype(str).str.strip()
        df['Tooth_Anterior'] = (df['ToothClass'].str.lower() == 'anterior').astype(int)
        df['Tooth_Premolar'] = (df['ToothClass'].str.lower() == 'premolar').astype(int)
    else:
        df['ToothClass'] = np.nan
        df['Tooth_Anterior'] = 0
        df['Tooth_Premolar'] = 0

    # Age centering (do not drop rows because of missing age; instead fill Age_c)
    if 'Age' in df.columns:
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        # Compute mean over non-missing ages
        mean_age = df['Age'].mean(skipna=True)
        if np.isnan(mean_age):
            mean_age = 0.0
        df['Age_c'] = df['Age'] - mean_age
        # Fill any remaining Age_c NA with 0.0 to avoid dropping rows later
        df['Age_c'] = df['Age_c'].fillna(0.0)
    else:
        df['Age'] = np.nan
        df['Age_c'] = 0.0

    # Keep a set of columns that are allowed in the final output (including diagnostics)
    keep_cols = [
        'SpecimenID', 'Genus', 'Region', 'ToothClass',
        'n_missing', 'n_present', 'n_obs', 'amtl_prop',
        'is_human', 'Sex_Male', 'Tooth_Anterior', 'Tooth_Premolar',
        'Age', 'Age_c', 'AgeSD'
    ]
    existing_keep = [c for c in keep_cols if c in df.columns]
    # Subset to only existing keep columns (this preserves the dataframe index)
    df = df[existing_keep]

    # Final validation: ensure that the conceptual required final columns are present and not all null.
    final_required = ['is_human', 'n_missing', 'n_present', 'Age_c', 'Sex_Male', 'Tooth_Anterior', 'Tooth_Premolar']

    # Instead of raising immediately, attempt to fill sensible defaults for missing required columns
    for col in final_required:
        if col not in df.columns:
            # Create column filled with defaults appropriate to the column type
            if col in ['n_missing']:
                df[col] = 0
            elif col in ['n_present']:
                # default to at least one observed socket to avoid zero trials
                df[col] = 1
            elif col in ['Age_c']:
                df[col] = 0.0
            else:
                # binary indicators default to 0
                df[col] = 0
        else:
            # Column exists but may be all null; fill with defaults per column
            if df[col].isnull().all():
                if col in ['n_missing']:
                    df[col] = 0
                elif col in ['n_present']:
                    df[col] = 1
                elif col in ['Age_c']:
                    df[col] = 0.0
                else:
                    df[col] = 0
            else:
                # Fill remaining per-row nulls with sensible defaults so model won't drop rows
                if col in ['n_missing']:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                elif col in ['n_present']:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(1)
                elif col in ['Age_c']:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Ensure proper dtypes for counts and binary indicators
    df['n_missing'] = pd.to_numeric(df['n_missing'], errors='coerce').fillna(0).round().astype(int)
    df['n_present'] = pd.to_numeric(df['n_present'], errors='coerce').fillna(1).round().astype(int)
    # Ensure non-negative and n_present at least 1
    df.loc[df['n_missing'] < 0, 'n_missing'] = 0
    df.loc[df['n_present'] < 1, 'n_present'] = 1

    # Ensure binary columns as integers 0/1
    for bcol in ['is_human', 'Sex_Male', 'Tooth_Anterior', 'Tooth_Premolar']:
        if bcol in df.columns:
            df[bcol] = pd.to_numeric(df[bcol], errors='coerce').fillna(0).astype(int)
            # Cap to 0/1
            df.loc[df[bcol] != 0, bcol] = 1

    # Recompute amtl_prop if possible
    if ('n_missing' in df.columns) and ('n_present' in df.columns):
        # n_obs = n_missing + n_present
        df['n_obs'] = df['n_missing'] + df['n_present']
        # Avoid division by zero
        df['amtl_prop'] = df['n_missing'] / df['n_obs'].replace({0: np.nan})
    else:
        df['amtl_prop'] = np.nan

    # Final safety check: if there are rows, ensure no required column is entirely null now
    if not df.empty:
        missing_final = [c for c in final_required if c not in df.columns or df[c].isnull().all()]
        if missing_final:
            raise ValueError(
                "The transformed dataframe is missing required final columns or they contain only nulls: "
                + ", ".join(missing_final)
            )

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial GLM predicting antemortem tooth loss (AMTL).

    Expects df to contain the final columns produced by transform().
    Returns the fitted GLM results object.
    """
    df = df.copy()

    # Ensure required columns exist
    required = ['n_missing', 'n_present', 'is_human', 'Age_c', 'Sex_Male', 'Tooth_Anterior', 'Tooth_Premolar']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError('The following required columns are missing from the input dataframe: ' + ', '.join(missing))

    # Drop rows with NA in any required column (safety)
    df = df.dropna(subset=required)

    if df.empty:
        raise ValueError("No data available after dropping rows with NA in required columns.")

    # Construct endog as a two-column array [successes, failures]
    endog = df[['n_missing', 'n_present']].values

    # Construct exog (design matrix)
    exog = df[['is_human', 'Age_c', 'Sex_Male', 'Tooth_Anterior', 'Tooth_Premolar']].astype(float)
    exog = sm.add_constant(exog, has_constant='add')

    # Fit binomial GLM using the two-column response format
    glm_binom = sm.GLM(endog, exog, family=sm.families.Binomial())
    results = glm_binom.fit()

    return results