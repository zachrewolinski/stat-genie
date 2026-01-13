from typing import Any
import re
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for binomial GLM.

    Returns a dataframe containing exactly the required final columns:
      ['Missing','Sockets','Missing_proportion','Genus','Age','Age_c',
       'AgeUncertainty','Sex','ToothClass','SpecimenID','Region']

    This function is robust to a variety of raw column names by selecting
    the first matching candidate for each required variable. If some of the
    less-essential candidate columns (AgeUncertainty, Sex, Region) are not
    present in the input, the function will create these columns filled with
    NaNs so that the final dataframe always contains the required columns.
    """
    df = df.copy()

    # Candidate input column names for each required output column.
    # The first candidate found in df.columns (case-insensitive and punctuation-insensitive) will be used.
    candidates = {
        'ToothClass': ['ToothClass', 'tooth_class', 'Tooth_Class', 'toothclass', 'feature1', 'Tooth'],
        'SpecimenID': ['SpecimenID', 'specimen_id', 'specimen', 'id', 'feature2'],
        'Missing': ['Missing', 'missing', 'AMTL', 'MissingCount', 'missing_teeth', 'feature3'],
        'Sockets': ['Sockets', 'sockets', 'SocketCount', 'SocketsCount', 'observable_sockets', 'feature4'],
        'Age': ['Age', 'age', 'EstimatedAge', 'estimated_age', 'feature5'],
        'AgeUncertainty': ['AgeUncertainty', 'age_uncertainty', 'Age_SD', 'age_sd', 'feature6'],
        'Sex': ['Sex', 'sex', 'sex_est', 'Sex_estimate', 'feature7', 'gender'],
        'Genus': ['Genus', 'genus', 'Taxon', 'taxon', 'feature8'],
        'Region': ['Region', 'region', 'Region_origin', 'region_origin', 'feature9']
    }

    def _normalize(name: Any) -> str:
        # Lowercase, strip, remove non-alphanumeric characters for robust matching
        s = str(name).lower().strip()
        return re.sub(r'[^a-z0-9]', '', s)

    # Map normalized actual column names to the real column name
    normalized_map = {_normalize(c): c for c in df.columns}

    found_map = {}  # maps existing column name -> desired output name
    missing_targets = []

    for target, cand_list in candidates.items():
        found = None
        for cand in cand_list:
            norm_cand = _normalize(cand)
            # Exact normalized match
            if norm_cand in normalized_map:
                found = normalized_map[norm_cand]
                break
            # Otherwise, try substring matches where the candidate is contained within the actual column name.
            # Do NOT match the reverse (actual contained within candidate) to avoid mapping short generic names
            # like 'age' to 'ageuncertainty'.
            for actual_norm, actual_orig in normalized_map.items():
                if norm_cand and (norm_cand in actual_norm):
                    found = actual_orig
                    break
            if found is not None:
                break

        if found is None:
            missing_targets.append((target, cand_list))
        else:
            # Ensure we don't map the same source column to multiple targets (ambiguous)
            if found in found_map:
                # If already mapped to same target, ignore; otherwise ambiguous
                if found_map[found] != target:
                    raise KeyError(
                        f"Ambiguous mapping: input column '{found}' would map to multiple targets "
                        f"('{found_map[found]}' and '{target}')."
                    )
            found_map[found] = target

    # For any required final targets that were not found in the input, create columns filled with NaN.
    # This ensures the final dataframe always contains the exact required column names.
    if missing_targets:
        for target, _ in missing_targets:
            # Only create the column if it doesn't already exist
            if target not in df.columns:
                df[target] = np.nan

    # Rename discovered columns to the exact required column names
    df = df.rename(columns=found_map)

    # Now ensure all required columns are present (defensive)
    final_required = ['Missing', 'Sockets', 'Age', 'AgeUncertainty', 'Sex', 'Genus', 'ToothClass', 'SpecimenID', 'Region']
    missing_after = [c for c in final_required if c not in df.columns]
    if missing_after:
        raise KeyError(f"After renaming/creation, the dataframe is still missing required columns: {missing_after}")

    # Convert types and clean values
    # Numeric conversions
    df['Missing'] = pd.to_numeric(df['Missing'], errors='coerce')
    df['Sockets'] = pd.to_numeric(df['Sockets'], errors='coerce')
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['AgeUncertainty'] = pd.to_numeric(df['AgeUncertainty'], errors='coerce')

    # Sex: try to coerce to numeric 0-1 if possible; otherwise attempt common string mappings
    # Preserve original values for mapping attempt
    sex_original = df['Sex'].copy()
    df['Sex'] = pd.to_numeric(df['Sex'], errors='coerce')
    if df['Sex'].isna().any():
        sex_map = {
            'm': 1.0, 'male': 1.0, 'man': 1.0,
            'f': 0.0, 'female': 0.0, 'woman': 0.0,
            '0': 0.0, '1': 1.0
        }

        def map_sex_val(v):
            if pd.isna(v):
                return np.nan
            # If v is numeric type already, return float
            if isinstance(v, (int, float, np.floating, np.integer)):
                return float(v)
            s = str(v).strip().lower()
            s_norm = re.sub(r'[^a-z0-9]', '', s)
            return sex_map.get(s, sex_map.get(s_norm, np.nan))

        # Apply mapping on the original (string) representation to maximize match chances
        mapped = sex_original.astype(object).apply(map_sex_val)
        df['Sex'] = df['Sex'].combine_first(mapped)

    # Drop rows with missing key modeling variables (do NOT require Sex or AgeUncertainty or Region here;
    # Sex may be missing for some observations and will be handled by imputation in model())
    # Important: do this before converting categorical columns to str so that NaNs are respected.
    df = df.dropna(subset=['Missing', 'Sockets', 'Age', 'Genus', 'ToothClass'])

    # Remove invalid socket counts (must be positive)
    df = df[df['Sockets'] > 0]

    # Compute Missing_proportion and guard numeric bounds
    df['Missing_proportion'] = df['Missing'] / df['Sockets']
    df['Missing_proportion'] = df['Missing_proportion'].clip(lower=0.0, upper=1.0)

    # Center Age
    # Use mean of remaining (non-missing) Age values
    age_mean = df['Age'].mean()
    df['Age_c'] = df['Age'] - age_mean

    # Now convert ID and categorical columns to strings and clean them
    df['SpecimenID'] = df['SpecimenID'].astype(str)
    df['ToothClass'] = df['ToothClass'].astype(str)
    df['Genus'] = df['Genus'].astype(str)
    df['Region'] = df['Region'].astype(str)

    # Clean categorical levels
    df['Genus'] = df['Genus'].str.strip()
    df['ToothClass'] = df['ToothClass'].str.strip().str.title()
    df['Region'] = df['Region'].str.strip()

    # Standardize some known genus labels (non-exhaustive and conservative)
    df['Genus'] = df['Genus'].replace({'Homo': 'Homo sapiens'})

    # Final column order (must match the contract)
    final_cols = ['Missing', 'Sockets', 'Missing_proportion', 'Genus', 'Age', 'Age_c',
                  'AgeUncertainty', 'Sex', 'ToothClass', 'SpecimenID', 'Region']
    df = df[final_cols].copy()

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial (logit) GLM for proportion of missing teeth with Sockets as the binomial denominator.

    Model formula:
      Missing_proportion ~ C(Genus) + Age_c + Sex + C(ToothClass)

    Uses cluster-robust standard errors clustered on SpecimenID.
    Returns the fitted results object (with cluster-robust covariance if available).
    """
    df = df.copy()

    # Verify required columns exist (presence only; missing values will be handled by patsy/statsmodels)
    required = ['Missing_proportion', 'Sockets', 'Genus', 'Age_c', 'Sex', 'ToothClass', 'SpecimenID']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Dataframe passed to model() is missing required columns: {missing}")

    if df.shape[0] == 0:
        raise ValueError("Dataframe passed to model() is empty; cannot fit model.")

    formula = 'Missing_proportion ~ C(Genus) + Age_c + Sex + C(ToothClass)'

    # Drop rows with missing values in the strictly required variables for the response and non-optional predictors.
    # Note: Sex is included in the model formula but may have missing values in the transformed data.
    # To avoid dropping many observations, impute missing Sex values with the mean Sex (or 0.5 if none observed).
    vars_in_formula_required = ['Missing_proportion', 'Genus', 'Age_c', 'ToothClass']
    df_for_model = df.dropna(subset=vars_in_formula_required).copy()

    if df_for_model.shape[0] == 0:
        raise ValueError("No observations remain after dropping rows with NA in required variables used by the model formula; cannot fit model.")

    # Ensure Sex is numeric and impute missing Sex values so patsy/statsmodels do not drop those rows.
    # Impute with the mean of observed Sex values; if no observed values, use 0.5 as a neutral midpoint.
    df_for_model['Sex'] = pd.to_numeric(df_for_model['Sex'], errors='coerce')
    if df_for_model['Sex'].isna().any():
        sex_mean = df_for_model['Sex'].mean()
        if pd.isna(sex_mean):
            fill_sex = 0.5
        else:
            fill_sex = sex_mean
        df_for_model['Sex'] = df_for_model['Sex'].fillna(fill_sex)

    # Build design matrices using patsy
    try:
        y, X = dmatrices(formula, df_for_model, return_type='dataframe')
    except Exception as e:
        raise RuntimeError(f"Failed to construct design matrices from formula '{formula}': {e}")

    if y.shape[0] == 0:
        raise ValueError("No observations remain after applying the model formula and dropping missing values; cannot fit model.")
    if X.shape[1] == 0:
        raise ValueError("No predictors in design matrix for the formula; cannot fit model.")

    # Align weights and cluster groups to the rows actually used by patsy (X.index)
    weights = df_for_model.loc[X.index, 'Sockets']
    cluster_groups = df_for_model.loc[X.index, 'SpecimenID']

    # Convert y to 1d array (proportion)
    y_endog = y.iloc[:, 0]

    # Fit GLM with binomial family using the design matrices
    # Use sm.GLM directly with the constructed matrices to avoid re-parsing formula and to ensure alignment.
    model_glm = sm.GLM(y_endog, X, family=sm.families.Binomial(), var_weights=weights)
    res = model_glm.fit()

    # Attempt cluster-robust covariance by SpecimenID
    try:
        res_clust = res.get_robustcov_results(cov_type='cluster', groups=cluster_groups)
    except Exception:
        res_clust = res

    print(res_clust.summary())
    return res_clust