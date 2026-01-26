from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/add_features_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Fair (affairs) dataset into a dataframe ready for modeling.

    Outputs (columns created/kept):
      - affairs: original count-coded variable (kept as provided)
      - any_affair: binary indicator (1 if affairs > 0 else 0)
      - children_binary: 1 if 'children' == 'yes', 0 if 'children' == 'no'
      - gender_male: 1 if gender == 'male', 0 if 'female'
      - children_x_male: interaction children_binary * gender_male
      - age_c, yearsmarried_c, religiousness_c, rating_c: centered continuous controls
      - education, occupation: kept as-is

    The function also drops rows with missing values for any variable used in the models.
    """
    df = df.copy()

    # Basic required columns check (will raise KeyError if missing)
    required_cols = ['affairs', 'children', 'gender', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    missing_required = [c for c in required_cols if c not in df.columns]
    if len(missing_required) > 0:
        raise KeyError(f"Missing required columns in input dataframe: {missing_required}")

    # Drop rows missing the core variables (we need 'affairs', 'children', 'gender' at minimum)
    df = df.dropna(subset=['affairs', 'children', 'gender'])

    # Recode children to binary (1=yes, 0=no). Be robust to capitalization/whitespace.
    df['children_binary'] = (
        df['children'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
    )

    # If values other than yes/no exist, they become NaN above; keep them as NaN so we'll drop later

    # Recode gender to binary male indicator (male=1, female=0)
    df['gender_male'] = (
        df['gender'].astype(str).str.strip().str.lower().map({'male': 1, 'female': 0})
    )

    # Derived dependent binary: any extramarital affair
    df['any_affair'] = (df['affairs'] > 0).astype(int)

    # Interaction term for moderation test
    df['children_x_male'] = df['children_binary'] * df['gender_male']

    # Center continuous covariates to improve estimation stability and interpretation
    for col in ['age', 'yearsmarried', 'religiousness', 'rating']:
        # If column exists and is numeric, center it; otherwise leave NaN which will be dropped
        if col in df.columns:
            df[col + '_c'] = df[col] - df[col].mean()
        else:
            df[col + '_c'] = np.nan

    # Ensure education and occupation are numeric (they are coded numerically in schema)
    # If not numeric, attempt coercion
    df['education'] = pd.to_numeric(df['education'], errors='coerce')
    df['occupation'] = pd.to_numeric(df['occupation'], errors='coerce')

    # Final column list that the model will use (these must be non-missing)
    model_cols = [
        'affairs', 'any_affair', 'children_binary', 'gender_male', 'children_x_male',
        'age_c', 'yearsmarried_c', 'religiousness_c', 'education', 'occupation', 'rating_c'
    ]

    # Drop rows with missing values in any of these columns
    df = df.dropna(subset=model_cols)

    # Convert to appropriate dtypes
    df['children_binary'] = df['children_binary'].astype(int)
    df['gender_male'] = df['gender_male'].astype(int)
    df['any_affair'] = df['any_affair'].astype(int)
    df['affairs'] = pd.to_numeric(df['affairs'], errors='coerce')

    # Re-check: drop rows where affairs could not be coerced
    df = df.dropna(subset=['affairs'])

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit two models to evaluate the effect of having children on extramarital affairs:
      1) Negative binomial (GLM) for the count outcome 'affairs' (primary analysis)
      2) Logistic regression for the binary outcome 'any_affair' (sensitivity / complementary analysis)

    Both models include the same covariates and a children x gender interaction to test moderation by gender.

    Returns a dict with keys 'neg_binom' and 'logit' containing the fitted results objects with robust (HC3) SEs applied where possible.
    """
    import statsmodels.formula.api as smf

    # Ensure required columns exist in df
    required_model_cols = [
        'affairs', 'any_affair', 'children_binary', 'gender_male', 'children_x_male',
        'age_c', 'yearsmarried_c', 'religiousness_c', 'education', 'occupation', 'rating_c'
    ]
    missing = [c for c in required_model_cols if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"Missing columns required for modeling: {missing}")

    # Formula used in both models
    formula = (
        'children_binary + gender_male + children_x_male + '
        'age_c + yearsmarried_c + religiousness_c + education + occupation + rating_c'
    )

    # Negative binomial model for the count of affairs
    # Use GLM with NegativeBinomial family (log link implied) for overdispersed counts
    nb_formula = 'affairs ~ ' + formula
    nb_model = smf.glm(formula=nb_formula, data=df, family=sm.families.NegativeBinomial()).fit()
    # Convert to robust covariance (HC3) results for inference
    try:
        nb_robust = nb_model.get_robustcov_results(cov_type='HC3')
    except Exception:
        # If robust conversion fails, fall back to the original fit
        nb_robust = nb_model

    # Logistic regression for any_affair (binary outcome)
    logit_formula = 'any_affair ~ ' + formula
    logit_model = smf.logit(formula=logit_formula, data=df).fit(disp=False)
    try:
        logit_robust = logit_model.get_robustcov_results(cov_type='HC3')
    except Exception:
        logit_robust = logit_model

    # Return both fitted result objects so the caller can inspect summaries, coefficients, CIs, etc.
    return {
        'neg_binom': nb_robust,
        'logit': logit_robust
    }


