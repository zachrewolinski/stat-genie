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
    # Make a copy to avoid modifying original
    df = df.copy()

    # Required raw columns from input dataset
    # feature6: beauty rating (numeric, mean shifted to zero in original)
    # feature7: evaluation score (1-5)
    # feature3: age
    # feature4: gender ('male'/'female')
    # feature2: minority ('yes'/'no')
    # feature5: single-credit course ('single'/'more')
    # feature8: course level ('lower'/'upper')
    # feature9: native english ('yes'/'no')
    # feature10: tenure track ('yes'/'no')
    # feature11: number responded
    # feature12: enrollment
    # feature13: instructor id

    # Drop rows missing key variables to ensure model estimation uses complete cases
    required = ['feature6','feature7','feature3','feature4','feature2',
                'feature5','feature8','feature9','feature10','feature11','feature12','feature13']
    df = df.dropna(subset=required)

    # Create analysis-ready columns
    df['Beauty'] = pd.to_numeric(df['feature6'], errors='coerce')
    df['EvalScore'] = pd.to_numeric(df['feature7'], errors='coerce')
    df['Age'] = pd.to_numeric(df['feature3'], errors='coerce')

    # Map categorical controls to binary indicators (explicit mapping to avoid unexpected categories)
    df['Minority'] = df['feature2'].map({'yes': 1, 'no': 0})
    df['Male'] = df['feature4'].map({'male': 1, 'female': 0})
    df['SingleCourse'] = df['feature5'].map({'single': 1, 'more': 0})
    df['UpperCourse'] = df['feature8'].map({'upper': 1, 'lower': 0})
    df['NativeEnglish'] = df['feature9'].map({'yes': 1, 'no': 0})
    df['OnTenureTrack'] = df['feature10'].map({'yes': 1, 'no': 0})

    df['NumResponded'] = pd.to_numeric(df['feature11'], errors='coerce')
    df['Enrollment'] = pd.to_numeric(df['feature12'], errors='coerce')
    df['InstructorID'] = pd.to_numeric(df['feature13'], errors='coerce')

    # Drop any rows that became NA after coercion
    df = df.dropna(subset=['Beauty', 'EvalScore', 'Age', 'NumResponded', 'Enrollment', 'InstructorID'])

    # Standardize beauty for interpretable coefficient (per-SD effect)
    df['Beauty_z'] = (df['Beauty'] - df['Beauty'].mean()) / df['Beauty'].std(ddof=0)

    # Log transforms for class size variables (helps with skewness and interpretable elasticity-like effects)
    # Enrollment has a minimum > 0 in this dataset; use natural log
    df['LogEnrollment'] = np.log(df['Enrollment'])
    df['LogNumResponded'] = np.log(df['NumResponded'])

    # Keep only columns needed for modeling (but do not remove original raw fields in case user wants them)
    model_cols = ['Beauty_z', 'EvalScore', 'Age', 'Male', 'Minority', 'SingleCourse', 'UpperCourse',
                  'NativeEnglish', 'OnTenureTrack', 'NumResponded', 'Enrollment', 'LogEnrollment',
                  'LogNumResponded', 'InstructorID']

    # Return dataframe that includes model columns (and original columns remain available)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    import statsmodels.formula.api as smf

    # Specify linear model: EvalScore on standardized Beauty and controls
    # We include log transforms of enrollment and respondents to control for class size effects
    formula = (
        'EvalScore ~ Beauty_z + Age + Male + Minority + SingleCourse + '
        'UpperCourse + NativeEnglish + OnTenureTrack + LogEnrollment + LogNumResponded'
    )

    # Fit OLS
    ols_mod = smf.ols(formula, data=df).fit()

    # Compute cluster-robust standard errors clustered at the instructor level
    # This adjusts inference for multiple courses per instructor
    results = ols_mod.get_robustcov_results(cov_type='cluster', groups=df['InstructorID'])

    # Print a concise summary and return results object for further inspection
    print(results.summary())
    return results


