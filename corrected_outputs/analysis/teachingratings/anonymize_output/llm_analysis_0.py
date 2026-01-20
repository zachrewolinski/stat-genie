from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/teachingratings/anonymize_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Hamermesh classroom dataset into a dataframe ready for modeling.

    Inputs (column names from raw):
    - feature2: minority ('no'/'yes')
    - feature3: age (numeric)
    - feature4: gender ('male'/'female')
    - feature5: single-credit elective ('single'/'more')
    - feature6: beauty rating (numeric, mean-shifted)
    - feature7: teaching evaluation (1-5)
    - feature8: course level ('lower'/'upper')
    - feature9: native english ('no'/'yes')
    - feature10: tenure track ('no'/'yes')
    - feature11: number participated (numeric)
    - feature12: number enrolled (numeric)
    - feature13: instructor id (numeric)

    Output columns (used in modeling):
    - Beauty, Beauty_z, TeachingEval, Age, Age_z, Gender_male, Minority,
      SingleCredit, UpperDivision, NativeEnglish, TenureTrack, LogClassSize,
      LogEnrollment, InstructorID
    """
    df = df.copy()

    # Rename raw columns to descriptive names
    rename_map = {
        'feature2': 'Minority_raw',
        'feature3': 'Age_raw',
        'feature4': 'Gender_raw',
        'feature5': 'SingleCredit_raw',
        'feature6': 'Beauty_raw',
        'feature7': 'TeachingEval',
        'feature8': 'CourseLevel_raw',
        'feature9': 'NativeEnglish_raw',
        'feature10': 'TenureTrack_raw',
        'feature11': 'NumParticipated_raw',
        'feature12': 'NumEnrolled_raw',
        'feature13': 'InstructorID'
    }
    df = df.rename(columns=rename_map)

    # Keep only rows with non-missing DV and IV
    df = df.dropna(subset=['TeachingEval', 'Beauty_raw'])

    # Coerce numeric columns
    df['TeachingEval'] = pd.to_numeric(df['TeachingEval'], errors='coerce')
    df['Beauty'] = pd.to_numeric(df['Beauty_raw'], errors='coerce')
    df['Age'] = pd.to_numeric(df['Age_raw'], errors='coerce')
    df['NumParticipated'] = pd.to_numeric(df['NumParticipated_raw'], errors='coerce')
    df['NumEnrolled'] = pd.to_numeric(df['NumEnrolled_raw'], errors='coerce')

    # Drop rows that became NA after coercion for key vars
    df = df.dropna(subset=['TeachingEval', 'Beauty', 'Age', 'NumParticipated', 'NumEnrolled', 'InstructorID'])

    # Binary / indicator variables from categorical text fields
    # feature2 (Minority): 'yes' means belongs to minority (non-Caucasian)
    df['Minority'] = df['Minority_raw'].astype(str).str.lower().map(lambda x: 1 if x == 'yes' else 0)

    # feature4 (Gender): 'male'/'female' -> Gender_male = 1 if male
    df['Gender_male'] = df['Gender_raw'].astype(str).str.lower().map(lambda x: 1 if x == 'male' else 0)

    # feature5 (Single credit elective): 'single' -> 1
    df['SingleCredit'] = df['SingleCredit_raw'].astype(str).str.lower().map(lambda x: 1 if x == 'single' else 0)

    # feature8 (Course level): 'upper' -> 1
    df['UpperDivision'] = df['CourseLevel_raw'].astype(str).str.lower().map(lambda x: 1 if x == 'upper' else 0)

    # feature9 (Native English): 'yes' -> 1
    df['NativeEnglish'] = df['NativeEnglish_raw'].astype(str).str.lower().map(lambda x: 1 if x == 'yes' else 0)

    # feature10 (Tenure track): 'yes' -> 1
    df['TenureTrack'] = df['TenureTrack_raw'].astype(str).str.lower().map(lambda x: 1 if x == 'yes' else 0)

    # Numeric transformations for scale and skew handling
    df['LogClassSize'] = np.log(df['NumParticipated'] + 1.0)
    df['LogEnrollment'] = np.log(df['NumEnrolled'] + 1.0)

    # Standardize continuous predictors for interpretability
    df['Beauty_z'] = (df['Beauty'] - df['Beauty'].mean()) / (df['Beauty'].std(ddof=0) if df['Beauty'].std(ddof=0) != 0 else 1)
    df['Age_z'] = (df['Age'] - df['Age'].mean()) / (df['Age'].std(ddof=0) if df['Age'].std(ddof=0) != 0 else 1)

    # Ensure InstructorID is integer/grouping factor
    df['InstructorID'] = pd.to_numeric(df['InstructorID'], errors='coerce')
    df = df.dropna(subset=['InstructorID'])
    # Convert to integer groups if possible
    try:
        df['InstructorID'] = df['InstructorID'].astype(int)
    except Exception:
        df['InstructorID'] = df['InstructorID'].astype('category').cat.codes

    # Final selected columns for modeling
    final_cols = [
        'Beauty', 'Beauty_z', 'TeachingEval', 'Age', 'Age_z', 'Gender_male', 'Minority',
        'SingleCredit', 'UpperDivision', 'NativeEnglish', 'TenureTrack', 'LogClassSize',
        'LogEnrollment', 'InstructorID'
    ]

    return df[final_cols].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a linear mixed-effects model predicting TeachingEval from standardized Beauty
    controlling for instructor and course covariates, with a random intercept for InstructorID.

    Model:
    TeachingEval_ij = beta0 + beta1*Beauty_z_ij + beta2*Age_z_ij + beta3*Gender_male_ij + ... + u_j + e_ij
    where u_j ~ N(0, sigma_u^2) is the instructor random intercept.

    Returns the fitted MixedLMResults object.
    """
    # Required columns
    required = [
        'TeachingEval', 'Beauty_z', 'Age_z', 'Gender_male', 'Minority',
        'SingleCredit', 'UpperDivision', 'NativeEnglish', 'TenureTrack',
        'LogClassSize', 'LogEnrollment', 'InstructorID'
    ]
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Transformed dataframe is missing required columns: {missing}")

    # Prepare exogenous matrix (fixed effects)
    exog_vars = [
        'Beauty_z', 'Age_z', 'Gender_male', 'Minority',
        'SingleCredit', 'UpperDivision', 'NativeEnglish', 'TenureTrack',
        'LogClassSize', 'LogEnrollment'
    ]
    exog = df[exog_vars].astype(float)
    exog = sm.add_constant(exog, has_constant='add')

    endog = df['TeachingEval'].astype(float)
    groups = df['InstructorID']

    # Fit mixed linear model with random intercept for instructor
    # Use REML=False for likelihood-based comparison if needed; method set to 'lbfgs' for reliability
    model = sm.MixedLM(endog, exog, groups=groups)
    try:
        result = model.fit(reml=False, method='lbfgs')
    except Exception:
        # fallback to default fit if lbfgs fails
        result = model.fit(reml=False)

    # Return the fitted model object (MixedLMResults) so the caller can inspect params, summary, etc.
    return result


