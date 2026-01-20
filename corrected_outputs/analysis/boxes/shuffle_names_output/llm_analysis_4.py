from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm


df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/boxes/shuffle_names_output/boxes.csv')


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to the analysis-ready dataframe.

    Produces the following columns required by the model:
      - ChoseMajority: binary outcome (1 if majority option chosen, 0 otherwise)
      - AgeYears: numeric age in years (from 'culture' column in provided schema)
      - Age_c: centered age (AgeYears - mean(AgeYears)) used in the model
      - CultureID: site/culture identifier (from 'y') as integer categorical id
      - IsMale: 1 if gender == 2 (per schema), 0 otherwise
      - MajorityDemonstratedFirst: binary indicator from 'age' column (per schema this encodes whether majority was shown first)
    """
    df = df.copy()

    # Ensure required columns exist; if not, this will raise a KeyError which signals schema mismatch
    required_cols = ['majority_first', 'culture', 'age', 'gender', 'y']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Input dataframe is missing required columns: {missing}")

    # Drop rows with missing critical raw values first
    df = df.dropna(subset=required_cols)

    # Coerce to numeric using helper columns to avoid pandas nullable dtypes
    df['_majority_first_num'] = pd.to_numeric(df['majority_first'], errors='coerce')
    df['AgeYears'] = pd.to_numeric(df['culture'], errors='coerce')
    df['_MajorityDemonstratedFirst_num'] = pd.to_numeric(df['age'], errors='coerce')
    df['_gender_num'] = pd.to_numeric(df['gender'], errors='coerce')
    df['_y_num'] = pd.to_numeric(df['y'], errors='coerce')

    # Drop any rows that failed numeric coercion
    numeric_cols = ['_majority_first_num', 'AgeYears', '_MajorityDemonstratedFirst_num', '_gender_num', '_y_num']
    df = df.dropna(subset=numeric_cols)

    # Convert numeric helper columns to native numpy dtypes (int64/float64)
    df['_majority_first_num'] = df['_majority_first_num'].astype(int)
    df['AgeYears'] = df['AgeYears'].astype(float)
    df['_MajorityDemonstratedFirst_num'] = df['_MajorityDemonstratedFirst_num'].astype(int)
    df['_gender_num'] = df['_gender_num'].astype(int)
    df['_y_num'] = df['_y_num'].astype(int)

    # Create final model columns using the required exact column names
    # MajorityDemonstratedFirst comes from 'age' per provided schema mapping
    df['MajorityDemonstratedFirst'] = df['_MajorityDemonstratedFirst_num'].astype(int)
    # IsMale: 1 if gender == 2, else 0
    df['IsMale'] = (df['_gender_num'] == 2).astype(int)
    # CultureID: integer site id
    df['CultureID'] = df['_y_num'].astype(int)
    # ChoseMajority: 1 if majority option chosen (schema: 2 = majority option), 0 otherwise
    df['ChoseMajority'] = (df['_majority_first_num'] == 2).astype(int)

    # Filter to plausible age range (per dataset description ages 4-14).
    df = df[(df['AgeYears'] >= 4) & (df['AgeYears'] <= 14)]

    # Keep only rows with valid majority_first choices (1,2,3)
    df = df[df['_majority_first_num'].isin([1, 2, 3])]

    # Center age for modeling interactions
    df['Age_c'] = df['AgeYears'] - df['AgeYears'].mean()

    # Final dataframe with exactly the columns used in the model
    out_cols = ['ChoseMajority', 'AgeYears', 'Age_c', 'CultureID', 'IsMale', 'MajorityDemonstratedFirst']
    # Ensure columns exist (in case of unexpected issues)
    for c in out_cols:
        if c not in df.columns:
            raise KeyError(f"Expected column {c} not found after transformation")

    # Return only the required columns (drop helper columns)
    return df[out_cols].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (GLM, binomial family) predicting the probability a child chooses the majority option.

    The primary predictor is centered age (Age_c). CultureID is included as a categorical moderator of the age effect
    via an interaction Age_c * C(CultureID). Controls include child sex (IsMale) and whether the majority was demonstrated first.

    Returns the fitted GLMResults object from statsmodels.
    """
    import statsmodels.formula.api as smf

    # Ensure required columns are present
    req = ['ChoseMajority', 'Age_c', 'CultureID', 'IsMale', 'MajorityDemonstratedFirst']
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise KeyError(f"Transformed dataframe is missing required columns for modeling: {missing}")

    # Work on a copy to avoid modifying caller's dataframe
    df = df.copy()

    # Ensure CultureID is a native integer dtype before converting to categorical for patsy/statsmodels
    if not np.issubdtype(df['CultureID'].dtype, np.integer):
        df['CultureID'] = pd.to_numeric(df['CultureID'], errors='coerce').astype(int)

    # Convert CultureID to categorical for formula; model will internally dummy-code it.
    df['CultureID'] = df['CultureID'].astype('category')

    # Formula: main effect of age, culture as categorical moderator, and an interaction between age and culture.
    # Controls: IsMale and MajorityDemonstratedFirst
    formula = 'ChoseMajority ~ Age_c * C(CultureID) + IsMale + MajorityDemonstratedFirst'

    # Fit GLM with binomial family (logistic regression).
    model_fit = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Return the fitted model object
    return model_fit