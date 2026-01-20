from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/affairs/anonymize_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw Fair (Psychology Today) survey dataframe into analysis-ready frame.

    Expected input columns: feature1..feature10 as described in schema.
    Produces at least the following columns used in modeling:
      - AffairFreq: numeric frequency (from feature2)
      - HasChildren: binary (1 yes, 0 no) (from feature6)
      - IsFemale: binary (1 female, 0 male) (from feature3)
      - Age (from feature4), YearsMarried (feature5), Religiosity (feature7),
        Education (feature8), Occupation (feature9), MaritalSatisfaction (feature10)
    """
    df = df.copy()

    # Rename the raw feature columns to meaningful names for clarity
    rename_map = {
        'feature2': 'AffairFreq',
        'feature3': 'Gender',
        'feature4': 'Age',
        'feature5': 'YearsMarried',
        'feature6': 'Children',
        'feature7': 'Religiosity',
        'feature8': 'Education',
        'feature9': 'Occupation',
        'feature10': 'MaritalSatisfaction'
    }
    df = df.rename(columns=rename_map)

    # Ensure affair frequency is numeric. The dataset uses numeric codes (0,1,2,3,7,12, ...)
    df['AffairFreq'] = pd.to_numeric(df['AffairFreq'], errors='coerce')

    # Standardize Children column to binary indicator HasChildren
    # Accept common string values 'yes'/'no' (case-insensitive). If already numeric (0/1), keep.
    if 'Children' in df.columns:
        # map strings to numeric
        df['Children_str'] = df['Children'].astype(str).str.strip().str.lower()
        df['HasChildren'] = df['Children_str'].map({'yes': 1, 'no': 0})
        # if mapping produced NaN but original values are 0/1 numeric, use them
        numeric_children = pd.to_numeric(df['Children'], errors='coerce')
        df.loc[df['HasChildren'].isna() & numeric_children.notna(), 'HasChildren'] = numeric_children
        df.drop(columns=['Children_str'], inplace=True)
    else:
        df['HasChildren'] = np.nan

    # Gender -> IsFemale (1 female, 0 male)
    df['Gender_str'] = df['Gender'].astype(str).str.strip().str.lower()
    df['IsFemale'] = df['Gender_str'].map({'female': 1, 'male': 0})
    # If gender coded differently (e.g., 'f'/'m' or numeric), attempt numeric fallback
    numeric_gender = pd.to_numeric(df['Gender'], errors='coerce')
    df.loc[df['IsFemale'].isna() & numeric_gender.notna(), 'IsFemale'] = numeric_gender
    df.drop(columns=['Gender_str'], inplace=True)

    # Convert control columns to numeric where appropriate
    for col in ['Age', 'YearsMarried', 'Religiosity', 'Education', 'Occupation', 'MaritalSatisfaction']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan

    # Drop rows with missing essential variables for the analysis
    required_cols = ['AffairFreq', 'HasChildren', 'IsFemale', 'Age', 'YearsMarried',
                     'Religiosity', 'Education', 'Occupation', 'MaritalSatisfaction']
    df = df.dropna(subset=required_cols)

    # Ensure integer type for affair frequency where appropriate
    # (keep as numeric; counts can be floats from coercion but model accepts numeric)
    df['AffairFreq'] = df['AffairFreq'].astype(float)

    # Final dataframe returned includes all model columns (and keeps other original columns)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a count model appropriate for an outcome with many zeros and overdispersion.

    We use a zero-inflated negative binomial model (ZINB) with the same covariates
    in the count and inflation parts. The main coefficient of interest is for HasChildren.

    Returns the fitted results object.
    """
    import statsmodels.api as sm
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

    # Ensure required columns are present
    exog_vars = ['HasChildren', 'IsFemale', 'Age', 'YearsMarried',
                 'Religiosity', 'Education', 'Occupation', 'MaritalSatisfaction']
    missing = [c for c in exog_vars + ['AffairFreq'] if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Dataframe is missing required columns for modeling: {missing}")

    # Exogenous matrix (include constant)
    exog = sm.add_constant(df[exog_vars].astype(float))
    endog = df['AffairFreq'].astype(float)

    # Fit Zero-Inflated Negative Binomial model
    # Use same exog for zero-inflation part so control variables explain both the inflation and count
    zinb = ZeroInflatedNegativeBinomialP(endog, exog, exog_infl=exog, inflation='logit')

    # Fit the model; capture the results
    try:
        res = zinb.fit(method='newton', maxiter=100, disp=0)
    except Exception:
        # fallback to default fit if newton fails
        res = zinb.fit(disp=0)

    # Return the results object (caller can inspect res.summary())
    return res


