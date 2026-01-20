from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/affairs/shuffle_names_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and create analysis-ready columns for the question: Does having children decrease engagement in extramarital affairs?

    Input dataframe uses the provided original column names (which are inconsistently labeled in the schema).
    This function will:
      - Create AffairFreq from the original 'education' column (which actually encodes affair frequency).
      - Create HasChildren from the original 'age' column (which actually encodes presence of children: 'yes'/'no').
      - Create IsFemale from the original 'children' column (which actually encodes gender: 'female'/'male').
      - Map other control variables from their appropriate (mismatched) columns.
      - Drop rows missing the key variables used in modeling.

    Returns a dataframe containing the exact columns used in the model.
    """
    df = df.copy()

    # Create dependent variable: Affair frequency (original column 'education' contains the affairs frequency codes)
    df['AffairFreq'] = pd.to_numeric(df.get('education'), errors='coerce')

    # Independent variable: presence of children (original 'age' column actually encodes 'yes'/'no' for children)
    # Normalize strings and map to binary
    df['HasChildren'] = df.get('age').astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})

    # Control: gender (original 'children' column actually contains 'male'/'female')
    df['IsFemale'] = df.get('children').astype(str).str.strip().str.lower().map({'female': 1, 'male': 0})

    # Control: Age (approximate age coding is stored in 'rating')
    df['Age'] = pd.to_numeric(df.get('rating'), errors='coerce')

    # Control: Years married (in this dataset, the 'gender' column contains years married codes)
    df['YearsMarried'] = pd.to_numeric(df.get('gender'), errors='coerce')

    # Control: Education level (in this dataset, the 'affairs' column contains education coding)
    df['EducationLevel'] = pd.to_numeric(df.get('affairs'), errors='coerce')

    # Control: Religiosity
    df['Religiosity'] = pd.to_numeric(df.get('religiousness'), errors='coerce')

    # Keep only rows with non-missing DV and IV; we will drop rows with missing key controls later in modeling
    df = df.dropna(subset=['AffairFreq', 'HasChildren'])

    # Cast to appropriate dtypes
    df['HasChildren'] = df['HasChildren'].astype('Int64')
    df['IsFemale'] = df['IsFemale'].astype('Float64')
    df['Age'] = df['Age'].astype('Float64')
    df['YearsMarried'] = df['YearsMarried'].astype('Float64')
    df['EducationLevel'] = df['EducationLevel'].astype('Float64')
    df['Religiosity'] = df['Religiosity'].astype('Float64')
    df['AffairFreq'] = df['AffairFreq'].astype('Float64')

    # Return only the columns that will be used in the model
    return df[['AffairFreq', 'HasChildren', 'IsFemale', 'Age', 'YearsMarried', 'EducationLevel', 'Religiosity']]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a count regression model for AffairFreq on HasChildren controlling for demographic covariates.

    We use a Negative Binomial GLM to account for count-like outcome and likely overdispersion.
    The model specification:
      AffairFreq ~ HasChildren + IsFemale + Age + YearsMarried + EducationLevel + Religiosity

    The function returns the fitted results object.
    """
    import statsmodels.api as sm

    # Work on a copy and drop rows with missing values in any predictor or outcome
    data = df.copy()
    data = data.dropna(subset=['AffairFreq', 'HasChildren', 'IsFemale', 'Age', 'YearsMarried', 'EducationLevel', 'Religiosity'])

    # Prepare design matrices
    y = data['AffairFreq'].astype(float)
    X = data[['HasChildren', 'IsFemale', 'Age', 'YearsMarried', 'EducationLevel', 'Religiosity']].astype(float)
    X = sm.add_constant(X, has_constant='add')

    # Fit Negative Binomial GLM (robust covariance)
    model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
    results = model.fit(cov_type='HC3')

    # Print a brief summary and return the results object
    print(results.summary())
    return results


