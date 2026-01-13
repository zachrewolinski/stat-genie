from typing import Any, Dict
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a raw dataframe into the FINAL dataframe required by the model.

    Required final columns (must be present in the returned dataframe):
      - 'SizeRatio' (log(SizeFocal / SizeOther))
      - 'LocationAdvantage' (DistOtherCenter - DistFocalCenter)
      - 'FocalWin' (0/1 integer outcome)
      - 'MalesDiff' (MalesFocal - MalesOther)
      - 'FemalesDiff' (FemalesFocal - FemalesOther)

    This function will attempt to rename common raw feature column names into the
    required descriptive names. It supports case-insensitive matching of common
    aliases for each required field. If required columns are still missing after
    the renaming attempt, a KeyError is raised.
    """
    df = df.copy()

    # Define canonical target column names (must not be changed)
    required_cols = [
        'FocalID', 'OtherID', 'DyadID', 'FocalWin',
        'DistFocalCenter', 'DistOtherCenter',
        'SizeFocal', 'SizeOther',
        'MalesFocal', 'MalesOther',
        'FemalesFocal', 'FemalesOther'
    ]

    # Common aliases (lowercase) mapped to the canonical target names.
    # This list includes the variants observed in the dataset that raised the error.
    alias_to_target: Dict[str, str] = {
        # Identifiers
        'focal': 'FocalID',
        'focalid': 'FocalID',
        'feature1': 'FocalID',
        'id_focal': 'FocalID',
        'group_focal': 'FocalID',

        'other': 'OtherID',
        'otherid': 'OtherID',
        'feature2': 'OtherID',
        'id_other': 'OtherID',
        'group_other': 'OtherID',

        'dyad': 'DyadID',
        'dyadid': 'DyadID',
        'feature3': 'DyadID',
        'pair_id': 'DyadID',
        'pair': 'DyadID',

        # Outcome
        'win': 'FocalWin',
        'focalwin': 'FocalWin',
        'result': 'FocalWin',
        'outcome': 'FocalWin',
        'feature4': 'FocalWin',

        # Distances
        'dist_focal': 'DistFocalCenter',
        'distfocalcenter': 'DistFocalCenter',
        'dist_focal_center': 'DistFocalCenter',
        'dist_focalcenter': 'DistFocalCenter',
        'feature5': 'DistFocalCenter',

        'dist_other': 'DistOtherCenter',
        'distothercenter': 'DistOtherCenter',
        'dist_other_center': 'DistOtherCenter',
        'dist_othercenter': 'DistOtherCenter',
        'feature6': 'DistOtherCenter',

        # Sizes / counts
        'n_focal': 'SizeFocal',
        'size_focal': 'SizeFocal',
        'sizefocal': 'SizeFocal',
        'n_foc': 'SizeFocal',
        'n_f': 'SizeFocal',
        'feature7': 'SizeFocal',

        'n_other': 'SizeOther',
        'size_other': 'SizeOther',
        'sizeother': 'SizeOther',
        'n_oth': 'SizeOther',
        'n_o': 'SizeOther',
        'feature8': 'SizeOther',

        # Males
        'm_focal': 'MalesFocal',
        'males_focal': 'MalesFocal',
        'malesfocal': 'MalesFocal',
        'feature9': 'MalesFocal',

        'm_other': 'MalesOther',
        'males_other': 'MalesOther',
        'malesother': 'MalesOther',
        'feature10': 'MalesOther',

        # Females
        'f_focal': 'FemalesFocal',
        'females_focal': 'FemalesFocal',
        'femalesfocal': 'FemalesFocal',
        'feature11': 'FemalesFocal',

        'f_other': 'FemalesOther',
        'females_other': 'FemalesOther',
        'femalesother': 'FemalesOther',
        'feature12': 'FemalesOther',
    }

    # Build rename map by inspecting existing dataframe columns.
    cols = list(df.columns)
    rename_map: Dict[str, str] = {}
    # For each column present, check if it matches any alias (case-insensitive).
    for col in cols:
        # If the column already matches one of the required target names, do nothing.
        if col in required_cols:
            continue
        lower = col.lower()
        if lower in alias_to_target:
            target = alias_to_target[lower]
            # Do not overwrite if the target already exists in the dataframe.
            if target in df.columns:
                continue
            # Map this raw column name to the canonical target.
            rename_map[col] = target

    if rename_map:
        df = df.rename(columns=rename_map)

    # Verify all necessary raw/descriptive columns are present before proceeding.
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"transform: missing required input columns after renaming: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Drop rows with missing values in any of the required input columns
    df = df.dropna(subset=required_cols).copy()

    # Ensure binary outcome is integer 0/1
    # If FocalWin is not numeric, try to coerce; this will raise if values cannot be converted.
    df['FocalWin'] = pd.to_numeric(df['FocalWin'], errors='raise').astype(int)

    # Compute relative size measures
    # SizeRatio: log(SizeFocal / SizeOther). Add a tiny constant for numerical safety.
    df['SizeRatio'] = np.log((df['SizeFocal'].astype(float) + 1e-6) / (df['SizeOther'].astype(float) + 1e-6))
    # SizeDiff (diagnostic helper - allowed as internal helper)
    df['SizeDiff'] = df['SizeFocal'] - df['SizeOther']

    # Composition differences (controls)
    df['MalesDiff'] = df['MalesFocal'] - df['MalesOther']
    df['FemalesDiff'] = df['FemalesFocal'] - df['FemalesOther']

    # Location advantage: DistOtherCenter - DistFocalCenter
    df['LocationAdvantage'] = df['DistOtherCenter'] - df['DistFocalCenter']

    # Optional categorical version of contest location for descriptive analyses (helper)
    df['ContestLocation'] = pd.cut(
        df['LocationAdvantage'],
        bins=[-np.inf, -50, 50, np.inf],
        labels=['OtherHome', 'Neutral', 'FocalHome']
    )

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    # Final check: ensure the final dataframe contains the conceptual variable columns
    final_required = ['SizeRatio', 'LocationAdvantage', 'FocalWin', 'MalesDiff', 'FemalesDiff', 'DyadID']
    missing_final = [c for c in final_required if c not in df.columns]
    if missing_final:
        raise KeyError(
            f"transform: missing required final columns: {missing_final}. "
            f"Available columns: {list(df.columns)}"
        )

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit the specified logistic regression model on the transformed dataframe.

    Model formula:
      FocalWin ~ SizeRatio * LocationAdvantage + MalesDiff + FemalesDiff

    Cluster-robust standard errors are requested clustered by 'DyadID' when possible.
    """
    # Ensure required columns are present
    required_for_model = ['FocalWin', 'SizeRatio', 'LocationAdvantage', 'MalesDiff', 'FemalesDiff', 'DyadID']
    missing = [c for c in required_for_model if c not in df.columns]
    if missing:
        raise KeyError(f"model: input dataframe is missing required columns: {missing}")

    formula = 'FocalWin ~ SizeRatio * LocationAdvantage + MalesDiff + FemalesDiff'

    # Fit logistic regression
    model_fit = smf.logit(formula, data=df).fit(disp=False)

    # Try to obtain cluster-robust SEs clustered by DyadID
    try:
        results = model_fit.get_robustcov_results(cov_type='cluster', groups=df['DyadID'])
    except Exception:
        # Fallback to default results if clustering fails
        results = model_fit

    return results