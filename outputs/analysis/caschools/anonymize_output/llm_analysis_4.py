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
    Transforms the raw dataset into a dataframe with the columns required for modeling.

    Input expected columns (from provided schema):
      - feature6: Enrollment (total students)
      - feature7: Number of teachers (FTE)
      - feature14: Average reading score
      - feature15: Average math score
      - feature11: Expenditure per student
      - feature8: Percent qualifying for CalWorks
      - feature9: Percent qualifying for reduced-price lunch
      - feature13: Percent English learners
      - feature10: Number of computers
      - feature12: District average income (in 1,000s USD)
      - feature5: Grade span (categorical)

    Returns a dataframe that includes at least the columns named in the conceptual variables.
    """
    df = df.copy()

    # Ensure numeric types where expected
    numeric_cols = ['feature6','feature7','feature11','feature8','feature9','feature13','feature10','feature12','feature14','feature15']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Create StudentTeacherRatio (guard against division by zero)
    # If feature7 (teachers) is zero or missing, result will be NaN
    df['StudentTeacherRatio'] = np.where(df['feature7'] > 0, df['feature6'] / df['feature7'], np.nan)

    # Create average score from reading and math
    df['AvgScore'] = df[['feature14', 'feature15']].mean(axis=1)

    # Map controls into clear column names
    df['ExpenditurePerStudent'] = df['feature11']
    df['PercentCalWorks'] = df['feature8']
    df['PercentReducedLunch'] = df['feature9']
    df['PctEnglishLearners'] = df['feature13']
    df['NumComputers'] = df['feature10']
    df['DistrictIncomeK'] = df['feature12']

    # Log of enrollment to capture nonlinear size effects (add 1 to avoid log(0))
    df['LogEnrollment'] = np.log(df['feature6'] + 1)

    # Grade span (categorical) - preserve original values but ensure dtype is category if present
    if 'feature5' in df.columns:
        df['GradeSpan'] = df['feature5'].astype('category')
    else:
        df['GradeSpan'] = pd.Categorical([None] * len(df))

    # Keep only rows with non-missing outcome and key independent variable
    required_for_model = [
        'AvgScore', 'StudentTeacherRatio', 'ExpenditurePerStudent', 'PercentCalWorks',
        'PercentReducedLunch', 'PctEnglishLearners', 'NumComputers', 'DistrictIncomeK', 'LogEnrollment'
    ]
    df = df.dropna(subset=required_for_model)

    # Optionally: remove extreme outliers in StudentTeacherRatio (e.g., implausible > 200)
    # We'll keep rows with reasonable ratios to avoid undue leverage. This threshold can be adjusted.
    df = df[df['StudentTeacherRatio'].abs() <= 200]

    # Final dataframe contains at least the columns required for the model
    final_cols = [
        'StudentTeacherRatio', 'AvgScore', 'ExpenditurePerStudent', 'PercentCalWorks',
        'PercentReducedLunch', 'PctEnglishLearners', 'NumComputers', 'DistrictIncomeK',
        'LogEnrollment', 'GradeSpan'
    ]

    # Return only the final columns (plus original ids if present) to keep modeling clean
    retain_cols = [c for c in (['feature1', 'feature2', 'feature3', 'feature4'] + final_cols) if c in df.columns]
    return df[retain_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fits an OLS model estimating the association between StudentTeacherRatio and AvgScore,
    controlling for district covariates. Returns the fitted statsmodels results object.

    Model specification:
      AvgScore ~ StudentTeacherRatio + ExpenditurePerStudent + PercentCalWorks
                 + PercentReducedLunch + PctEnglishLearners + NumComputers
                 + DistrictIncomeK + LogEnrollment + C(GradeSpan)

    Robust (HC3) standard errors are used to reduce sensitivity to heteroskedasticity.
    """
    import statsmodels.formula.api as smf

    # Ensure the dataframe contains the columns used in the formula
    formula = (
        'AvgScore ~ StudentTeacherRatio + ExpenditurePerStudent + PercentCalWorks '
        '+ PercentReducedLunch + PctEnglishLearners + NumComputers '
        '+ DistrictIncomeK + LogEnrollment + C(GradeSpan)'
    )

    # Fit OLS with robust standard errors
    model = smf.ols(formula, data=df).fit()

    # Compute robust covariance (HC3)
    try:
        robust_results = model.get_robustcov_results(cov_type='HC3')
    except Exception:
        # If robust covariance fails for any reason, fall back to standard results
        robust_results = model

    # Print brief summary and return the robust results object for downstream inspection
    print(robust_results.summary())
    return robust_results


