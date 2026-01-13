from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/shuffle_names_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy to avoid side-effects
    df = df.copy()

    # NOTES ON DATASET MAPPING (the provided schema has several mislabelings):
    # - The column 'education' in this dataset contains the reported extramarital frequency coding (0,1,2,3,7,12 etc.).
    # - The column 'affairs' contains the education level codes (9,12,14,16,17,18,20).
    # - The column 'age' contains the 'Are there children in the marriage?' factor ("yes"/"no").
    # - The column 'children' contains gender labels ("male"/"female").
    # - The column 'gender' contains years married coding (numeric).
    # We'll derive clean variables for analysis.

    # 1) AffairCount: use the column 'education' which in this file actually holds the extramarital frequency codes.
    # Coerce to numeric; keep original coded values as-is (they represent frequency categories used in Fair's dataset).
    df['AffairCount'] = pd.to_numeric(df['education'], errors='coerce')

    # 2) Binary indicator: AffairAny
    df['AffairAny'] = (df['AffairCount'].fillna(0) > 0).astype(int)

    # 3) HasChildren: derive from column 'age' which contains 'yes'/'no' per the provided schema
    # Normalize strings and map
    if 'age' in df.columns:
        df['HasChildren'] = df['age'].astype(str).str.strip().str.lower().map({'yes': 1, 'y': 1, 'no': 0, 'n': 0})
    else:
        df['HasChildren'] = np.nan

    # 4) Gender: from column 'children' (which contains 'male'/'female' in this dataset)
    if 'children' in df.columns:
        df['IsFemale'] = df['children'].astype(str).str.strip().str.lower().map({'female': 1, 'f': 1, 'male': 0, 'm': 0})
    else:
        df['IsFemale'] = np.nan

    # 5) Age in years: column 'rating' uses age midpoints; coerce to numeric and keep
    df['Age'] = pd.to_numeric(df['rating'], errors='coerce')

    # 6) EducationYears: column 'affairs' actually contains education codes in this dataset
    df['EducationYears'] = pd.to_numeric(df['affairs'], errors='coerce')

    # 7) Religiousness: keep as provided
    df['Religiousness'] = pd.to_numeric(df['religiousness'], errors='coerce')

    # 8) YearsMarried: use column 'gender' which the schema indicates contains years married coding
    df['YearsMarried'] = pd.to_numeric(df['gender'], errors='coerce')

    # 9) MarriageHappiness: use 'rownames' column which encodes self-rated marriage happiness
    df['MarriageHappiness'] = pd.to_numeric(df['rownames'], errors='coerce')

    # 10) Drop rows with missing key variables for our primary analysis
    # We require: AffairCount (DV) and HasChildren (IV) and at least basic controls Age and EducationYears.
    df = df.dropna(subset=['AffairCount', 'HasChildren', 'Age', 'EducationYears'])

    # 11) If IsFemale or other controls have missing values we keep row but will handle missingness in model by dropping there or imputing.
    # For convenience, ensure the important control columns exist (create if missing, filled with NaN)
    for col in ['IsFemale', 'Religiousness', 'YearsMarried', 'MarriageHappiness']:
        if col not in df.columns:
            df[col] = np.nan

    # 12) Final type enforcement
    df['AffairCount'] = pd.to_numeric(df['AffairCount'], errors='coerce')
    df['AffairAny'] = df['AffairAny'].astype(int)
    df['HasChildren'] = pd.to_numeric(df['HasChildren'], errors='coerce')
    df['IsFemale'] = pd.to_numeric(df['IsFemale'], errors='coerce')
    df['EducationYears'] = pd.to_numeric(df['EducationYears'], errors='coerce')
    df['Religiousness'] = pd.to_numeric(df['Religiousness'], errors='coerce')
    df['YearsMarried'] = pd.to_numeric(df['YearsMarried'], errors='coerce')
    df['MarriageHappiness'] = pd.to_numeric(df['MarriageHappiness'], errors='coerce')

    # 13) Return the transformed dataframe containing only the columns needed for modeling (plus originals kept implicitly)
    keep_cols = [
        'AffairCount', 'AffairAny', 'HasChildren', 'IsFemale', 'Age', 'EducationYears',
        'Religiousness', 'YearsMarried', 'MarriageHappiness'
    ]
    # Ensure we return all columns that may be needed; if some are missing because of earlier drops they will not error here
    existing_keep = [c for c in keep_cols if c in df.columns]
    return df[existing_keep]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    # df is the transformed dataframe produced by transform()
    # We'll run two models:
    # 1) Primary: logistic regression for the probability of any extramarital affair (AffairAny) ~ HasChildren + controls
    # 2) Robustness: negative binomial GLM for AffairCount ~ HasChildren + controls (appropriate for overdispersed counts)

    results = {}

    # Prepare modeling dataframe: drop rows with missing values in the variables used for each model
    # Define common predictors
    predictors = ['HasChildren', 'IsFemale', 'Age', 'EducationYears', 'Religiousness', 'YearsMarried', 'MarriageHappiness']
    # Keep only predictors that exist in the df
    predictors = [p for p in predictors if p in df.columns]

    # 1) Logistic regression (occurrence)
    if 'AffairAny' in df.columns:
        mod_df = df.dropna(subset=['AffairAny'] + predictors)
        if mod_df.shape[0] >= 10:
            X = mod_df[predictors]
            X = sm.add_constant(X, has_constant='add')
            y = mod_df['AffairAny']
            try:
                logit_model = sm.Logit(y, X).fit(disp=False)
                results['logit'] = logit_model
            except Exception as e:
                # fallback to GLM with binomial family if Logit has convergence issues
                try:
                    glm_binom = sm.GLM(y, X, family=sm.families.Binomial()).fit()
                    results['logit_glm'] = glm_binom
                except Exception as e2:
                    results['logit_error'] = str(e2)
        else:
            results['logit_error'] = 'Not enough observations after dropping missing for logistic model.'

    # 2) Negative binomial (count) for AffairCount as robustness
    if 'AffairCount' in df.columns:
        mod_df2 = df.dropna(subset=['AffairCount'] + predictors)
        if mod_df2.shape[0] >= 10:
            X2 = mod_df2[predictors]
            X2 = sm.add_constant(X2, has_constant='add')
            y2 = mod_df2['AffairCount']
            # If AffairCount has many zeros, NB is appropriate; use GLM NegativeBinomial family
            try:
                nb_model = sm.GLM(y2, X2, family=sm.families.NegativeBinomial()).fit()
                results['neg_binom'] = nb_model
            except Exception as e:
                # fallback to Poisson if NB fails
                try:
                    pois_model = sm.GLM(y2, X2, family=sm.families.Poisson()).fit()
                    results['poisson'] = pois_model
                except Exception as e2:
                    results['count_error'] = str(e2)
        else:
            results['count_error'] = 'Not enough observations after dropping missing for count model.'

    return results


