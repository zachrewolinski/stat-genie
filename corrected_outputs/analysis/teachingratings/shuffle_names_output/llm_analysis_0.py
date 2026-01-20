from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/teachingratings/shuffle_names_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy to avoid modifying original
    df = df.copy()

    # Map core variables to clearer names and coerce to numeric where appropriate
    # Dependent variable: evaluation score (original column 'tenure')
    df['EvalScore'] = pd.to_numeric(df['tenure'], errors='coerce')

    # Independent: continuous beauty/attractiveness score (original column 'prof')
    df['BeautyScore'] = pd.to_numeric(df['prof'], errors='coerce')

    # Controls: age, gender, class level, enrollment, number of raters, tenure-track, native English
    df['Age'] = pd.to_numeric(df['division'], errors='coerce')  # 'division' contains ages (29-73)

    # 'age' column in the provided schema contains gender labels (male/female)
    df['Gender'] = df['age'].astype(str).str.lower().str.strip()

    df['ClassLevel'] = df['students'].astype(str).str.lower().str.strip()  # 'lower' / 'upper'

    df['NumEnrolled'] = pd.to_numeric(df['credits'], errors='coerce')
    df['NRaters'] = pd.to_numeric(df['minority'], errors='coerce')

    # Tenure-track indicator: original 'eval' column contains 'yes'/'no' per schema
    df['TenureTrack'] = df['eval'].astype(str).str.lower().map(lambda x: 1 if x == 'yes' else 0)

    # Native English speaker indicator: original 'allstudents' contains 'yes'/'no'
    df['NativeEnglish'] = df['allstudents'].astype(str).str.lower().map(lambda x: 1 if x == 'yes' else 0)

    # Remove rows missing the main variables (DV or IV)
    df = df.dropna(subset=['EvalScore', 'BeautyScore'])

    # Standardize the beauty score (z-score)
    df['Beauty_z'] = (df['BeautyScore'] - df['BeautyScore'].mean()) / df['BeautyScore'].std(ddof=1)

    # Create a binary 'high-beauty' indicator (top quartile) for an alternative specification
    q75 = df['BeautyScore'].quantile(0.75)
    df['BeautyHigh'] = (df['BeautyScore'] >= q75).astype(int)

    # Create common dummies (drop_first=True to avoid multicollinearity)
    # Gender dummy (will produce 'Gender_male' if 'male' and 'female' are present)
    gender_dummies = pd.get_dummies(df['Gender'], prefix='Gender', drop_first=True)
    df = pd.concat([df, gender_dummies], axis=1)

    # Class level dummy: 'ClassLevel_upper' if 'upper' present
    class_dummies = pd.get_dummies(df['ClassLevel'], prefix='ClassLevel', drop_first=True)
    df = pd.concat([df, class_dummies], axis=1)

    # Ensure dummy columns exist even if a category is missing (set to 0)
    if 'Gender_male' not in df.columns:
        df['Gender_male'] = 0
    if 'ClassLevel_upper' not in df.columns:
        df['ClassLevel_upper'] = 0

    # Return the dataframe with the newly-created and required columns
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Make a defensive copy
    df = df.copy()

    # Ensure required columns exist (if not, create neutral defaults)
    for col in ['EvalScore', 'Beauty_z', 'BeautyHigh', 'Age', 'NumEnrolled', 'NRaters', 'TenureTrack', 'NativeEnglish', 'Gender_male', 'ClassLevel_upper']:
        if col not in df.columns:
            if col in ['BeautyHigh', 'TenureTrack', 'NativeEnglish', 'Gender_male', 'ClassLevel_upper']:
                df[col] = 0
            else:
                df[col] = pd.NA

    # Drop rows with missing values in model variables (after defaults applied)
    model_vars = ['EvalScore', 'Beauty_z', 'Age', 'NumEnrolled', 'NRaters', 'TenureTrack', 'NativeEnglish', 'Gender_male', 'ClassLevel_upper']
    df_mod = df.dropna(subset=model_vars)

    # Continuous beauty specification
    formula_cont = 'EvalScore ~ Beauty_z + Age + NumEnrolled + NRaters + TenureTrack + NativeEnglish + Gender_male + ClassLevel_upper'
    model_cont = smf.ols(formula_cont, data=df_mod).fit(cov_type='HC3')  # robust (HC3) SEs

    # Binary (top-quartile) beauty specification
    df_mod2 = df_mod.copy()
    # Ensure BeautyHigh exists in df_mod2
    if 'BeautyHigh' not in df_mod2.columns:
        df_mod2['BeautyHigh'] = 0
    formula_bin = 'EvalScore ~ BeautyHigh + Age + NumEnrolled + NRaters + TenureTrack + NativeEnglish + Gender_male + ClassLevel_upper'
    model_bin = smf.ols(formula_bin, data=df_mod2).fit(cov_type='HC3')

    # Return the fitted model results objects (user can call .summary() on them)
    results = {
        'continuous_beauty_model': model_cont,
        'binary_top_quartile_beauty_model': model_bin
    }
    return results


