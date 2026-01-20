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

    # Rename relevant columns to meaningful names based on schema
    rename_map = {
        'feature6': 'Enrollment',               # Total enrollment
        'feature7': 'NumTeachers',              # Number of teachers (FTE)
        'feature8': 'PercentCalWorks',          # Percent CalWorks
        'feature9': 'PercentReducedLunch',      # Percent reduced-price lunch
        'feature10': 'NumComputers',            # Number of computers
        'feature11': 'ExpenditurePerStudent',   # Expenditure per student
        'feature12': 'DistrictIncome_k',        # District average income (in 1,000s)
        'feature13': 'PercentEnglishLearners',  # Percent English learners
        'feature14': 'AvgReading',              # Average reading score
        'feature15': 'AvgMath',                 # Average math score
        'feature5': 'GradeSpan',                # Grade span factor (KK-06 / KK-08)
        'feature4': 'County'                    # County factor
    }
    df = df.rename(columns=rename_map)

    # Ensure numeric columns are numeric; coerce invalid entries to NaN
    num_cols = ['Enrollment', 'NumTeachers', 'PercentCalWorks', 'PercentReducedLunch',
                'NumComputers', 'ExpenditurePerStudent', 'DistrictIncome_k',
                'PercentEnglishLearners', 'AvgReading', 'AvgMath']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing core variables required for the analysis
    required = ['Enrollment', 'NumTeachers', 'AvgReading', 'AvgMath']
    df = df.dropna(subset=required)

    # Remove invalid or zero teacher counts to avoid division by zero
    df = df[df['NumTeachers'] > 0]

    # Compute student-teacher ratio (students per teacher)
    df['StudentTeacherRatio'] = df['Enrollment'] / df['NumTeachers']

    # Compute the dependent variable: average of reading and math scores
    df['AvgTestScore'] = (df['AvgReading'] + df['AvgMath']) / 2.0

    # Controls: keep percent variables as-is (already numeric), create binary grade-span indicator
    df['GradeSpan_KK08'] = (df['GradeSpan'] == 'KK-08').astype(int)

    # Log of enrollment (control for skew in district size)
    # Add a small constant if enrollment could be 0 (already filtered), but guard anyway
    df['LogEnrollment'] = np.log(df['Enrollment'].replace(0, np.nan))

    # Create county dummy variables for fixed effects (drop_first to avoid multicollinearity)
    if 'County' in df.columns:
        county_dummies = pd.get_dummies(df['County'].astype(str), prefix='County', drop_first=True)
        # Concatenate dummies to df
        df = pd.concat([df.reset_index(drop=True), county_dummies.reset_index(drop=True)], axis=1)

    # Final columns to keep (includes generated county dummies automatically if present)
    # We keep all derived and control columns so the modeling function can select features
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Build the regression model to estimate association between student-teacher ratio and test scores
    # Make a local copy
    data = df.copy()

    # Dependent variable
    y = data['AvgTestScore']

    # Start with main independent variable and baseline controls
    base_controls = [
        'StudentTeacherRatio',
        'PercentReducedLunch',
        'PercentEnglishLearners',
        'PercentCalWorks',
        'ExpenditurePerStudent',
        'NumComputers',
        'DistrictIncome_k',
        'GradeSpan_KK08',
        'LogEnrollment'
    ]

    # Add county dummies if present in dataframe (columns starting with 'County_')
    county_cols = [c for c in data.columns if c.startswith('County_')]
    X_cols = base_controls + county_cols

    # Keep only columns that actually exist in the dataframe
    X_cols = [c for c in X_cols if c in data.columns]

    X = data[X_cols].copy()

    # Drop rows where any X or y is missing
    model_df = pd.concat([y, X], axis=1).dropna()
    y_clean = model_df['AvgTestScore']
    X_clean = model_df.drop(columns=['AvgTestScore'])

    # Add constant for intercept
    X_clean = sm.add_constant(X_clean)

    # Fit OLS
    results = sm.OLS(y_clean, X_clean).fit()

    # Print summary (optional) and return fitted results object
    print(results.summary())
    return results


