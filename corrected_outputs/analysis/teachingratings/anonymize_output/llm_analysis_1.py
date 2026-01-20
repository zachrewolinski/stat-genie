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
    Transform the raw dataset into a dataframe with the columns needed for modeling.

    Final columns produced (and used by the model):
      - Eval: numeric teaching evaluation score (feature7)
      - Beauty: numeric beauty rating (feature6)
      - Beauty_sq: squared beauty term
      - is_female: 1 if female, 0 if male
      - Age: numeric age (feature3)
      - is_minority: 1 if minority (feature2 == 'yes') else 0
      - is_single_credit: 1 if single-credit elective (feature5 == 'single') else 0
      - upper_division: 1 if upper division (feature8 == 'upper') else 0
      - native_english: 1 if native English (feature9 == 'yes') else 0
      - tenure: 1 if on tenure track (feature10 == 'yes') else 0
      - class_size: number of students who participated in evaluations (feature11)
      - enrollment: number enrolled in course (feature12)
      - log_class_size: log( class_size ) (natural log)
      - log_enrollment: log( enrollment )
      - instructor_id: instructor identifier (feature13)

    The function coerces types, creates binary indicators, derives quadratic and log terms,
    and drops rows with missing values for variables used in the models.
    """
    df = df.copy()

    # Core variables (coerce to numeric where appropriate)
    df['Beauty'] = pd.to_numeric(df['feature6'], errors='coerce')
    df['Eval'] = pd.to_numeric(df['feature7'], errors='coerce')
    df['Age'] = pd.to_numeric(df['feature3'], errors='coerce')

    # Binary / categorical -> numeric encoding (map conservative: missing -> NaN)
    df['is_female'] = df['feature4'].astype(str).str.lower().map({'female': 1, 'male': 0})
    df['is_minority'] = df['feature2'].astype(str).str.lower().map({'yes': 1, 'no': 0})
    df['is_single_credit'] = df['feature5'].astype(str).str.lower().map({'single': 1, 'more': 0})
    df['upper_division'] = df['feature8'].astype(str).str.lower().map({'upper': 1, 'lower': 0})
    df['native_english'] = df['feature9'].astype(str).str.lower().map({'yes': 1, 'no': 0})
    df['tenure'] = df['feature10'].astype(str).str.lower().map({'yes': 1, 'no': 0})

    # Class size / enrollment and instructor id
    df['class_size'] = pd.to_numeric(df['feature11'], errors='coerce')
    df['enrollment'] = pd.to_numeric(df['feature12'], errors='coerce')
    df['instructor_id'] = df['feature13']

    # Derived variables
    df['Beauty_sq'] = df['Beauty'] ** 2

    # Log transforms: replace non-positive with NaN before log
    df.loc[df['class_size'] <= 0, 'class_size'] = np.nan
    df.loc[df['enrollment'] <= 0, 'enrollment'] = np.nan
    df['log_class_size'] = np.log(df['class_size'])
    df['log_enrollment'] = np.log(df['enrollment'])

    # Rows must have dependent and main independent variable
    required_cols = [
        'Eval', 'Beauty', 'Beauty_sq', 'is_female', 'Age', 'is_minority',
        'is_single_credit', 'upper_division', 'native_english', 'tenure',
        'log_class_size', 'log_enrollment', 'instructor_id'
    ]

    # Drop rows missing any required modeling column
    df = df.dropna(subset=required_cols)

    # Ensure types: integers for binary indicators
    int_cols = [
        'is_female', 'is_minority', 'is_single_credit', 'upper_division',
        'native_english', 'tenure'
    ]
    for c in int_cols:
        df[c] = df[c].astype(int)

    # instructor_id to string/categorical for fixed effects modeling
    df['instructor_id'] = df['instructor_id'].astype(str)

    # Keep only columns relevant for modeling (but don't drop original raw columns to preserve context)
    model_cols = [
        'Eval', 'Beauty', 'Beauty_sq', 'is_female', 'Age', 'is_minority',
        'is_single_credit', 'upper_division', 'native_english', 'tenure',
        'class_size', 'enrollment', 'log_class_size', 'log_enrollment', 'instructor_id'
    ]

    return df[model_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit models to estimate the impact of Beauty on Eval.

    Returns a dictionary with three fitted models (statsmodels RegressionResults):
      - 'ols': baseline OLS with controls
      - 'fe_instructor': OLS with instructor fixed effects (C(instructor_id))
      - 'interaction_gender': baseline model plus Beauty:is_female interaction

    All models use heteroskedasticity-robust (HC1) standard errors when fitting.
    """
    df = df.copy()

    # Ensure instructor treated as categorical for fixed effects
    df['instructor_id'] = df['instructor_id'].astype('category')

    base_formula = (
        'Eval ~ Beauty + Beauty_sq + is_female + Age + is_minority '
        '+ is_single_credit + upper_division + native_english + tenure '
        '+ log_class_size + log_enrollment'
    )

    # 1) Baseline OLS with controls
    model_ols = smf.ols(base_formula, data=df).fit(cov_type='HC1')

    # 2) Add instructor fixed effects via categorical instructor_id
    model_fe_instructor = smf.ols(base_formula + ' + C(instructor_id)', data=df).fit(cov_type='HC1')

    # 3) Interaction between Beauty and gender to test moderation by gender
    model_interaction = smf.ols(base_formula + ' + Beauty:is_female', data=df).fit(cov_type='HC1')

    # Return the fitted results objects for further inspection.
    return {
        'ols': model_ols,
        'fe_instructor': model_fe_instructor,
        'interaction_gender': model_interaction,
    }