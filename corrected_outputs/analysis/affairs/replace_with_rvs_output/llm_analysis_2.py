from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/affairs/replace_with_rvs_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Fair (Psychology Today) dataset into a cleaned dataframe ready for modeling.

    Produces the following important columns used in the model:
      - affairs (numeric count; kept from original)
      - HasChildren (binary: 1 if children == 'yes', else 0)
      - Female (binary: 1 if gender == 'female', else 0)
      - age, yearsmarried, religiousness, education, occupation, rating (cleaned/typed)

    Rows with missing values in any of these core columns are dropped.
    """
    # work on a copy
    df = df.copy()

    # Normalize string columns for safe matching
    if 'children' in df.columns:
        df['children'] = df['children'].astype(str).str.strip().str.lower()
    else:
        raise KeyError("Expected column 'children' in dataframe")

    if 'gender' in df.columns:
        df['gender'] = df['gender'].astype(str).str.strip().str.lower()
    else:
        raise KeyError("Expected column 'gender' in dataframe")

    # Create HasChildren binary (1 = yes, 0 = no). Treat anything not explicitly 'yes' as 0.
    df['HasChildren'] = np.where(df['children'].isin(['yes', 'y', 'true', '1']), 1, 0)

    # Create Female binary variable
    df['Female'] = np.where(df['gender'] == 'female', 1, 0)

    # Ensure affairs is numeric
    df['affairs'] = pd.to_numeric(df['affairs'], errors='coerce')

    # Ensure numeric control columns are numeric
    for col in ['age', 'yearsmarried', 'religiousness', 'education', 'rating']:
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' in dataframe")
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Occupation should be treated as categorical
    if 'occupation' not in df.columns:
        raise KeyError("Expected column 'occupation' in dataframe")
    # Keep occupation as-is but ensure consistent type
    df['occupation'] = df['occupation'].astype('category')

    # Drop rows with missing values in any columns used in the model
    required_cols = ['affairs', 'HasChildren', 'Female', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    df = df.dropna(subset=required_cols)

    # Reset index after filtering
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a Negative Binomial generalized linear model to estimate the association between
    having children and the count of extramarital affairs, controlling for demographic and
    marriage-related covariates.

    Model specification (primary):
      affairs ~ HasChildren + Female + age + yearsmarried + religiousness + education + C(occupation) + rating

    Rationale: affairs is a count variable with overdispersion (variance > mean). Negative
    Binomial (GLM) handles overdispersion better than Poisson. We include occupation as a
    categorical control using C(occupation) in the formula.

    The function returns the fitted results object (statsmodels GLMResults).
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Formula including occupation as categorical
    formula = 'affairs ~ HasChildren + Female + age + yearsmarried + religiousness + education + C(occupation) + rating'

    # Fit Negative Binomial GLM
    model = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial())
    results = model.fit()

    # Print a short summary for quick inspection (caller can inspect 'results' further)
    print(results.summary())

    return results


