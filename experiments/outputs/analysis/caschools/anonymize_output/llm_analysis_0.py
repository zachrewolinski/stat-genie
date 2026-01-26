from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/caschools/anonymize_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the dataframe used for modeling.

    Creates the following columns required by the model:
      - AvgTestScore: mean of feature14 (avg reading) and feature15 (avg math)
      - StudentTeacherRatio: feature6 (enrollment) / feature7 (teachers FTE)
      - ExpPerStudent: feature11 (expenditure per student)
      - PctReducedLunch: feature9
      - PctEngLearners: feature13
      - ComputersPerStudent: feature10 / feature6
      - DistrictIncome: feature12 (in 1,000 USD)
      - County: feature4 (kept as categorical)
      - GradeSpan: feature5 (kept as categorical)

    Drops rows with missing critical fields and avoids division-by-zero.
    """
    df = df.copy()

    # Ensure numeric columns are numeric
    numeric_cols = ['feature6', 'feature7', 'feature10', 'feature11', 'feature12', 'feature13', 'feature9', 'feature14', 'feature15']
    for c in numeric_cols:
        # coerce errors to NaN
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Create dependent variable: average of reading and math scores
    df['AvgTestScore'] = df[['feature14', 'feature15']].mean(axis=1)

    # Prevent division by zero for teachers; set to NaN if teachers missing or zero
    df.loc[df['feature7'] == 0, 'feature7'] = np.nan

    # Independent variable: student-teacher ratio
    df['StudentTeacherRatio'] = df['feature6'] / df['feature7']

    # Controls (rename/massage columns to the final names used in modeling)
    df['ExpPerStudent'] = df['feature11']
    df['PctReducedLunch'] = df['feature9']
    df['PctEngLearners'] = df['feature13']
    # Computers per student; if enrollment is zero or missing, result will be NaN
    df['ComputersPerStudent'] = df['feature10'] / df['feature6']
    df['DistrictIncome'] = df['feature12']

    # Categorical controls: keep original values, convert to string/object
    df['County'] = df['feature4'].astype('category')
    df['GradeSpan'] = df['feature5'].astype('category')

    # Drop rows missing DV or IV or key controls used in the regression
    required = ['AvgTestScore', 'StudentTeacherRatio', 'ExpPerStudent', 'PctReducedLunch', 'PctEngLearners']
    df = df.dropna(subset=required)

    # Optionally, remove extreme outliers for StudentTeacherRatio (e.g., > 200) which are likely data errors
    # but keep most natural variation. We'll cap extremely large ratios at the 99.9th percentile to reduce influence.
    if df['StudentTeacherRatio'].notna().sum() > 0:
        upper = df['StudentTeacherRatio'].quantile(0.999)
        if np.isfinite(upper) and upper > 0:
            df.loc[df['StudentTeacherRatio'] > upper, 'StudentTeacherRatio'] = upper

    # Reset index for modeling convenience
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS regression of AvgTestScore on StudentTeacherRatio with controls.

    Model specification:
      AvgTestScore ~ StudentTeacherRatio + ExpPerStudent + PctReducedLunch + PctEngLearners
                   + ComputersPerStudent + DistrictIncome + C(County) + C(GradeSpan)

    Uses heteroskedasticity-robust standard errors (HC3).

    Returns the fitted results (statsmodels RegressionResults wrapper).
    """
    import statsmodels.formula.api as smf

    # formula using the exact column names created in transform
    formula = ('AvgTestScore ~ StudentTeacherRatio + ExpPerStudent + PctReducedLunch '
               '+ PctEngLearners + ComputersPerStudent + DistrictIncome + C(County) + C(GradeSpan)')

    # Fit OLS
    model = smf.ols(formula=formula, data=df).fit(cov_type='HC3')

    # Print summary for quick inspection (can be removed in production)
    print(model.summary())

    return model


