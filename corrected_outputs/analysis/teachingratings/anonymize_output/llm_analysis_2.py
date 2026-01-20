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
    Transform raw dataset to variables used in modeling.
    Input: original dataframe with columns feature1..feature13 (see schema)
    Output: dataframe with columns used in the statistical model:
      - Beauty, Beauty_c
      - Eval
      - Female, Minority, Age, Age_c, SingleCredit, UpperDivision, NativeEnglish, TenureTrack
      - NumStudentsRated, LogNumStudents, Enrollment, LogEnrollment
      - InstructorID

    The function drops rows missing the key dependent/independent variables and required controls.
    """
    df = df.copy()

    # Ensure numeric columns are numeric and drop rows with missing outcome or main predictor
    df['feature6'] = pd.to_numeric(df['feature6'], errors='coerce')  # beauty rating
    df['feature7'] = pd.to_numeric(df['feature7'], errors='coerce')  # eval score
    df = df.dropna(subset=['feature6', 'feature7'])

    # Dependent and independent variables
    df['Beauty'] = df['feature6']
    df['Eval'] = df['feature7']

    # Controls: map textual categories to binaries and coerce numerics
    # feature2: minority ("yes"/"no")
    df['Minority'] = df['feature2'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
    # feature4: gender ("male"/"female") -> Female indicator
    df['Female'] = df['feature4'].astype(str).str.strip().str.lower().map({'female': 1, 'male': 0})
    # feature3: age
    df['Age'] = pd.to_numeric(df['feature3'], errors='coerce')
    # feature5: single-credit elective (samples show 'single' or 'more') -> treat 'single' as single-credit elective
    df['SingleCredit'] = df['feature5'].astype(str).str.strip().str.lower().map({'single': 1, 'more': 0})
    # feature8: course division ('upper'/'lower')
    df['UpperDivision'] = df['feature8'].astype(str).str.strip().str.lower().map({'upper': 1, 'lower': 0})
    # feature9: native english ('yes'/'no')
    df['NativeEnglish'] = df['feature9'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
    # feature10: tenure track ('yes'/'no')
    df['TenureTrack'] = df['feature10'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})

    # Numeric course-size controls
    df['NumStudentsRated'] = pd.to_numeric(df['feature11'], errors='coerce')
    df['Enrollment'] = pd.to_numeric(df['feature12'], errors='coerce')

    # Instructor identifier for fixed effects
    df['InstructorID'] = df['feature13']

    # Log transforms (clip at 1 to avoid log(0))
    df['LogNumStudents'] = np.log(df['NumStudentsRated'].clip(lower=1))
    df['LogEnrollment'] = np.log(df['Enrollment'].clip(lower=1))

    # Center continuous covariates to aid interpretation
    df['Beauty_c'] = df['Beauty'] - df['Beauty'].mean()
    df['Age_c'] = df['Age'] - df['Age'].mean()

    # Drop rows with remaining missing values in model columns
    required_cols = [
        'Beauty', 'Beauty_c', 'Eval', 'Female', 'Minority', 'Age', 'Age_c',
        'SingleCredit', 'UpperDivision', 'NativeEnglish', 'TenureTrack',
        'NumStudentsRated', 'LogNumStudents', 'Enrollment', 'LogEnrollment', 'InstructorID'
    ]
    df = df.dropna(subset=required_cols)

    # Ensure InstructorID is treated as categorical-friendly (keep as is; C(...) will be used in formula)
    df['InstructorID'] = df['InstructorID'].astype('category')

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS model estimating the effect of beauty on teaching evaluations.

    Model specification (main):
      Eval ~ Beauty_c + Female + Minority + Age_c + SingleCredit + UpperDivision
             + NativeEnglish + TenureTrack + LogNumStudents + LogEnrollment
             + Beauty_c:Female + C(InstructorID)

    - Uses heteroskedasticity-robust standard errors (HC3).
    - Returns the fitted results object (statsmodels.regression.linear_model.RegressionResultsWrapper).
    """
    # Formula includes an interaction between beauty and female (gender as moderator) and instructor fixed effects
    formula = (
        'Eval ~ Beauty_c + Female + Minority + Age_c + SingleCredit + UpperDivision '
        '+ NativeEnglish + TenureTrack + LogNumStudents + LogEnrollment '
        '+ Beauty_c:Female + C(InstructorID)'
    )

    results = smf.ols(formula, data=df).fit(cov_type='HC3')

    # The caller can inspect results.summary() or use results.params, results.bse, etc.
    return results