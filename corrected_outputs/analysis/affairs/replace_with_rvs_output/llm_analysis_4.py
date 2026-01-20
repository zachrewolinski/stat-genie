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
    # Work on a copy
    df = df.copy()

    # Required source columns (from schema):
    # 'affairs','children','gender','age','yearsmarried','religiousness','education','occupation','rating'
    # Drop rows missing any required raw inputs
    required_raw = ['affairs', 'children', 'gender', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    df = df.dropna(subset=required_raw)

    # Dependent variable: keep a numeric copy named 'Affairs'
    df['Affairs'] = pd.to_numeric(df['affairs'], errors='coerce')

    # Independent variable: HasChildren (1=yes, 0=no). Be robust to capitalization/whitespace.
    df['children'] = df['children'].astype(str)
    df['HasChildren'] = df['children'].str.strip().str.lower().map(lambda x: 1 if x in ['yes', 'y', 'true', '1'] else 0)

    # Moderator / control: Male indicator from 'gender'
    df['gender'] = df['gender'].astype(str)
    df['Male'] = df['gender'].str.strip().str.lower().map(lambda x: 1 if x == 'male' else 0)

    # Numeric controls: coerce to numeric and rename to descriptive final columns
    df['Age'] = pd.to_numeric(df['age'], errors='coerce')
    df['YearsMarried'] = pd.to_numeric(df['yearsmarried'], errors='coerce')
    df['Religiousness'] = pd.to_numeric(df['religiousness'], errors='coerce')
    df['Education'] = pd.to_numeric(df['education'], errors='coerce')
    df['Occupation'] = pd.to_numeric(df['occupation'], errors='coerce')
    df['Rating'] = pd.to_numeric(df['rating'], errors='coerce')

    # Interaction term for moderation test: HasChildren x Male
    df['HasChildren_Male'] = df['HasChildren'] * df['Male']

    # Drop any rows with NA in the model columns
    model_cols = ['Affairs', 'HasChildren', 'Male', 'HasChildren_Male', 'Age', 'YearsMarried', 'Religiousness', 'Education', 'Occupation', 'Rating']
    df = df.dropna(subset=model_cols)

    # Ensure integer types for binary columns
    df['HasChildren'] = df['HasChildren'].astype(int)
    df['Male'] = df['Male'].astype(int)
    df['HasChildren_Male'] = df['HasChildren_Male'].astype(int)

    # Final dataframe contains at least the columns listed in the conceptual model
    # Return the transformed dataframe
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit a count regression to estimate the association between having children and number of affairs.
    We use a Negative Binomial GLM (log link) to account for overdispersion in the count outcome.

    Model specification:
      Affairs ~ HasChildren + Male + HasChildren_Male + Age + YearsMarried + Religiousness + Education + Occupation + Rating

    The interaction term tests whether the association of children with affairs differs by gender.
    """
    df = df.copy()

    # Outcome and design matrix
    y = df['Affairs']
    X = df[['HasChildren', 'Male', 'HasChildren_Male', 'Age', 'YearsMarried', 'Religiousness', 'Education', 'Occupation', 'Rating']]
    X = sm.add_constant(X)

    # Fit Negative Binomial GLM (log link by default)
    model_nb = sm.GLM(y, X, family=sm.families.NegativeBinomial())
    results = model_nb.fit()

    # Print and return results (results contains params, pvalues, conf_int, etc.)
    print(results.summary())
    return results


