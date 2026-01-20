from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/caschools/anonymize_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Coerce numeric columns (use dataset field names from schema)
    df['Enrollment'] = pd.to_numeric(df['feature6'], errors='coerce')
    df['NumTeachers'] = pd.to_numeric(df['feature7'], errors='coerce')
    df['PercCalWorks'] = pd.to_numeric(df['feature8'], errors='coerce')
    df['PercReducedLunch'] = pd.to_numeric(df['feature9'], errors='coerce')
    df['NumComputers'] = pd.to_numeric(df['feature10'], errors='coerce')
    df['ExpenditurePerStudent'] = pd.to_numeric(df['feature11'], errors='coerce')
    df['IncomeK'] = pd.to_numeric(df['feature12'], errors='coerce')
    df['PercEngLearners'] = pd.to_numeric(df['feature13'], errors='coerce')
    df['ReadingScore'] = pd.to_numeric(df['feature14'], errors='coerce')
    df['MathScore'] = pd.to_numeric(df['feature15'], errors='coerce')

    # Categorical variables
    df['County'] = df['feature4'].astype('category')
    df['GradeSpan'] = df['feature5'].astype('category')

    # Remove rows with invalid or zero denominators
    df = df[(df['Enrollment'] > 0) & (df['NumTeachers'] > 0)]

    # Drop rows missing key outcome or denominator values
    df = df.dropna(subset=['ReadingScore', 'MathScore', 'Enrollment', 'NumTeachers'])

    # Compute primary variables
    df['StudentTeacherRatio'] = df['Enrollment'] / df['NumTeachers']
    df['AvgScore'] = df[['ReadingScore', 'MathScore']].mean(axis=1)
    df['ComputersPerStudent'] = df['NumComputers'] / df['Enrollment']
    df['LogEnrollment'] = np.log1p(df['Enrollment'])

    # Trim extreme StudentTeacherRatio outliers (optional conservative filter)
    # Keep observations within 4 SDs of the mean ratio to avoid single extreme districts dominating results
    ratio_mean = df['StudentTeacherRatio'].mean()
    ratio_std = df['StudentTeacherRatio'].std()
    if pd.notnull(ratio_mean) and pd.notnull(ratio_std) and ratio_std > 0:
        df = df[(df['StudentTeacherRatio'] - ratio_mean).abs() <= 4 * ratio_std]

    # Impute remaining missing control values with median (conservative simple approach)
    for col in ['PercCalWorks', 'PercReducedLunch', 'ExpenditurePerStudent', 'IncomeK', 'PercEngLearners', 'ComputersPerStudent']:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Final check: drop any rows missing the modeling columns
    model_cols = ['AvgScore', 'StudentTeacherRatio', 'ExpenditurePerStudent', 'IncomeK', 'PercCalWorks', 'PercReducedLunch', 'PercEngLearners', 'ComputersPerStudent', 'LogEnrollment', 'County', 'GradeSpan']
    df = df.dropna(subset=[c for c in model_cols if c in df.columns])

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    import statsmodels.formula.api as smf

    # Model specification:
    # AvgScore is regressed on StudentTeacherRatio (primary IV) and a set of controls.
    # County and GradeSpan are included as categorical fixed effects.
    formula = (
        'AvgScore ~ StudentTeacherRatio + ExpenditurePerStudent + IncomeK + '
        'PercCalWorks + PercReducedLunch + PercEngLearners + ComputersPerStudent + LogEnrollment + '
        'C(County) + C(GradeSpan)'
    )

    # Fit OLS with robust (HC3) standard errors to reduce influence of heteroskedasticity
    model = smf.ols(formula, data=df).fit(cov_type='HC3')

    # Return the fitted model object (caller can inspect summary(), params, conf_int(), etc.)
    return model


