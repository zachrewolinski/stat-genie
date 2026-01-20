from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset into analysis-ready dataframe.
    - Keeps and coerces relevant columns
    - Drops rows missing the essential vars
    - Creates centered beauty measure, log class-size controls, and an interaction term
    - Returns a dataframe containing only the columns used in the model
    """
    df = df.copy()

    # Ensure required raw columns exist in df
    required_raw = [
        'feature2','feature3','feature4','feature5','feature6','feature7',
        'feature8','feature9','feature10','feature11','feature12','feature13'
    ]
    missing = [c for c in required_raw if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f'Missing required input columns: {missing}')

    # Coerce numeric columns
    df['Beauty'] = pd.to_numeric(df['feature6'], errors='coerce')
    df['EvalScore'] = pd.to_numeric(df['feature7'], errors='coerce')
    df['Age'] = pd.to_numeric(df['feature3'], errors='coerce')
    df['NResponded'] = pd.to_numeric(df['feature11'], errors='coerce')
    df['NEnrolled'] = pd.to_numeric(df['feature12'], errors='coerce')

    # Instructor ID: coerce to numeric first (float), drop rows with missing, then convert to plain int dtype
    df['InstructorID'] = pd.to_numeric(df['feature13'], errors='coerce')

    # Map categorical flags to binaries (standardize text to lower case first)
    df['Gender_male'] = df['feature4'].astype(str).str.lower().map({'male': 1, 'female': 0})
    df['Minority'] = df['feature2'].astype(str).str.lower().map({'yes': 1, 'no': 0})
    df['SingleCredit'] = df['feature5'].astype(str).str.lower().map({'single': 1, 'more': 0})
    df['UpperDiv'] = df['feature8'].astype(str).str.lower().map({'upper': 1, 'lower': 0})
    df['NativeEnglish'] = df['feature9'].astype(str).str.lower().map({'yes': 1, 'no': 0})
    df['TenureTrack'] = df['feature10'].astype(str).str.lower().map({'yes': 1, 'no': 0})

    # Drop rows missing core variables
    core_cols = ['Beauty', 'EvalScore', 'Age', 'Gender_male', 'NResponded', 'NEnrolled', 'InstructorID']
    df = df.dropna(subset=core_cols)

    # Now it's safe to convert InstructorID to a numpy integer dtype (avoids pandas' nullable integer dtype)
    df['InstructorID'] = df['InstructorID'].astype(int)

    # Mean-center beauty for interpretability
    df['Beauty_centered'] = df['Beauty'] - df['Beauty'].mean()

    # Log transforms of counts (clip to avoid log(0))
    df['LogNResponded'] = np.log(df['NResponded'].clip(lower=1))
    df['LogNEnrolled'] = np.log(df['NEnrolled'].clip(lower=1))

    # Interaction to test heterogeneous effect by gender
    df['Beauty_x_Gender'] = df['Beauty_centered'] * df['Gender_male']

    # Return only the columns used in the analysis/model (keeps ordering clear)
    out_cols = [
        'Beauty', 'Beauty_centered', 'EvalScore', 'Age', 'Gender_male', 'Minority',
        'SingleCredit', 'UpperDiv', 'NativeEnglish', 'TenureTrack',
        'NResponded', 'NEnrolled', 'LogNResponded', 'LogNEnrolled',
        'InstructorID', 'Beauty_x_Gender'
    ]

    return df[out_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """ 
    Runs an OLS regression of teaching evaluations on beauty and controls.
    - Tests interaction between beauty and gender
    - Includes instructor fixed effects via C(InstructorID)
    - Uses cluster-robust standard errors clustered by InstructorID

    Returns the fitted results object.
    """
    # Ensure dataframe contains required columns
    required = [
        'EvalScore', 'Beauty_centered', 'Gender_male', 'Beauty_x_Gender', 'Age',
        'LogNResponded', 'LogNEnrolled', 'SingleCredit', 'UpperDiv', 'Minority',
        'NativeEnglish', 'TenureTrack', 'InstructorID'
    ]
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f'Missing required columns for modeling: {missing}')

    # Specify formula (instructor fixed effects via C(InstructorID))
    formula = (
        'EvalScore ~ Beauty_centered + Gender_male + Beauty_x_Gender + Age '
        '+ LogNResponded + LogNEnrolled + SingleCredit + UpperDiv + Minority + NativeEnglish + TenureTrack '
        '+ C(InstructorID)'
    )

    ols = smf.ols(formula=formula, data=df)

    # Fit with cluster-robust standard errors clustered at instructor level
    results = ols.fit(cov_type='cluster', cov_kwds={'groups': df['InstructorID']})

    # Print brief summary; return results for downstream inspection
    print(results.summary())
    return results