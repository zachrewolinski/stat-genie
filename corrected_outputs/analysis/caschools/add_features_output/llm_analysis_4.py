from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/caschools/add_features_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataset for modeling the relationship between student-teacher ratio and academic performance.

    Produces the following columns required by the model:
    - stu_teacher_ratio: students / teachers (students per teacher)
    - AvgScore: mean of 'read' and 'math'
    - computers_per_student: computer / students
    - log_expenditure: natural log of 'expenditure' (per student)
    - grades_KK08: indicator (1 if grades == 'KK-08', 0 otherwise)

    Drops rows with missing or problematic values for the key columns.
    """
    df = df.copy()

    # Keep only rows with the core variables present
    required = ['students', 'teachers', 'read', 'math']
    df = df.dropna(subset=required)

    # Remove impossible / zero teachers to avoid division by zero
    df = df[df['teachers'] > 0]

    # Compute student-teacher ratio (students per teacher)
    df['stu_teacher_ratio'] = df['students'] / df['teachers']

    # Dependent variable: average of reading and math scores
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Resource controls
    # computers per student (if students > 0 after previous filtering)
    df['computers_per_student'] = df['computer'] / df['students']

    # Log transform of expenditure per student; set NaN for nonpositive expenditures
    df['log_expenditure'] = np.where(df['expenditure'] > 0, np.log(df['expenditure']), np.nan)

    # Binary indicator for grade span KK-08 vs others (KK-06 expected)
    # ensure string comparison even if categories
    df['grades_KK08'] = (df['grades'].astype(str) == 'KK-08').astype(int)

    # Keep the socioeconomic and demographic controls using original column names
    # Columns: income, lunch, calworks, english, students, county

    # Drop rows with missing values in any of the model columns
    model_cols = [
        'stu_teacher_ratio', 'AvgScore', 'computers_per_student', 'log_expenditure',
        'income', 'lunch', 'calworks', 'english', 'students', 'grades_KK08', 'county'
    ]
    df = df.dropna(subset=model_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Runs an OLS regression of AvgScore on stu_teacher_ratio controlling for covariates.

    Formula:
      AvgScore ~ stu_teacher_ratio + log_expenditure + income + lunch + english
                 + computers_per_student + students + grades_KK08

    Returns the fitted statsmodels results object.
    """
    import statsmodels.formula.api as smf

    # Fit OLS model. Analysts may choose to add county fixed effects by uncommenting C(county).
    formula = (
        'AvgScore ~ stu_teacher_ratio + log_expenditure + income + lunch + english '
        '+ computers_per_student + students + grades_KK08'
    )

    # If users wish to include county fixed effects, they can change the formula to include + C(county)
    results = smf.ols(formula, data=df).fit()

    return results


