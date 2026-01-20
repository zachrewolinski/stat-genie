from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/teachingratings/add_features_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Hamermesh classroom dataset into a dataframe ready for modeling.

    Steps performed:
    - Make a defensive copy of the input.
    - Drop rows missing the core variables (beauty, eval).
    - Rename eval -> Eval (dependent variable) and ensure numeric type.
    - Standardize beauty to create beauty_z (mean 0, sd 1).
    - Create binary indicator controls from categorical fields: Female, Tenure, Minority, Native, UpperDivision, SingleCredit.
    - Create LogStudents = log(students) to reduce skew.
    - Drop any rows with missing values in the variables used in the model.
    - Ensure 'prof' is integer (will be used as a factor in the model).
    """
    df = df.copy()

    # Require core variables
    df = df.dropna(subset=['beauty', 'eval'])

    # Dependent variable: rename to Eval
    df['Eval'] = pd.to_numeric(df['eval'], errors='coerce')

    # Independent variable: standardized beauty
    df['beauty_z'] = (df['beauty'] - df['beauty'].mean()) / (df['beauty'].std(ddof=0) if df['beauty'].std(ddof=0) != 0 else 1.0)

    # Binary / indicator controls created from categorical columns (map known levels -> 0/1)
    df['Female'] = df['gender'].map({'female': 1, 'male': 0})
    df['Tenure'] = df['tenure'].map({'yes': 1, 'no': 0})
    df['Minority'] = df['minority'].map({'yes': 1, 'no': 0})
    df['Native'] = df['native'].map({'yes': 1, 'no': 0})
    df['UpperDivision'] = df['division'].map({'upper': 1, 'lower': 0})
    df['SingleCredit'] = df['credits'].map({'single': 1, 'more': 0})

    # Continuous controls / transforms
    # students should be positive; replace zeros (if any) with NaN to avoid -inf in log
    df['students'] = pd.to_numeric(df['students'], errors='coerce')
    df.loc[df['students'] <= 0, 'students'] = np.nan
    df['LogStudents'] = np.log(df['students'])

    # Ensure other numeric controls are numeric
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['religiousness'] = pd.to_numeric(df['religiousness'], errors='coerce')
    df['expenditure'] = pd.to_numeric(df['expenditure'], errors='coerce')

    # Ensure professor id is integer (for fixed effects / clustering)
    df['prof'] = pd.to_numeric(df['prof'], errors='coerce')

    # Drop rows missing any variable used in the model
    required_cols = [
        'beauty_z', 'Eval', 'Female', 'age', 'Tenure', 'Minority', 'Native',
        'UpperDivision', 'SingleCredit', 'LogStudents', 'religiousness', 'expenditure', 'prof'
    ]
    df = df.dropna(subset=required_cols)

    # Cast indicator columns to integer dtype (cleaner for modeling)
    for c in ['Female', 'Tenure', 'Minority', 'Native', 'UpperDivision', 'SingleCredit']:
        df[c] = df[c].astype(int)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> object:
    """
    Run an OLS regression of Eval on standardized beauty and controls, with professor fixed effects.

    Model specification:
      Eval ~ beauty_z + Female + age + Tenure + Minority + Native
             + UpperDivision + SingleCredit + LogStudents + religiousness + expenditure
             + C(prof)

    Returns:
      A statsmodels RegressionResults object with clustered standard errors by professor (if possible).
    """
    import statsmodels.formula.api as smf

    # Formula including professor fixed effects via C(prof)
    formula = (
        'Eval ~ beauty_z + Female + age + Tenure + Minority + Native '
        '+ UpperDivision + SingleCredit + LogStudents + religiousness + expenditure + C(prof)'
    )

    ols_mod = smf.ols(formula=formula, data=df)
    res = ols_mod.fit()

    # Try to compute clustered standard errors by professor id
    try:
        clustered = res.get_robustcov_results(cov_type='cluster', groups=df['prof'])
        return clustered
    except Exception:
        # If clustering fails, return the plain OLS results
        return res


