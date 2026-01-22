from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/anonymize_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Fair (Psychology Today) dataset into a cleaned dataframe with columns used in modeling.

    Expected original columns (per provided schema):
      - feature1: respondent id (kept as RespondentID)
      - feature2: frequency of extramarital intercourse in past year (0,1,2,3,7,12 etc.)
      - feature3: gender ("female" / "male")
      - feature4: age coding
      - feature5: years married
      - feature6: are there children in the marriage? ("yes"/"no")
      - feature7: religiosity (1-5)
      - feature8: education
      - feature9: occupation
      - feature10: marriage happiness (1-5)

    Returns a dataframe containing the following columns (exact names used in modeling):
      - RespondentID, AffairCount, AffairBinary, HasChildren, IsMale, Gender, Age,
        YearsMarried, Religiosity, Education, Occupation, MarriageHappiness
    """

    # Make a copy to avoid modifying original
    df = df.copy()

    # Rename columns to meaningful names
    rename_map = {
        'feature1': 'RespondentID',
        'feature2': 'AffairCount',
        'feature3': 'Gender',
        'feature4': 'Age',
        'feature5': 'YearsMarried',
        'feature6': 'HasChildren_raw',
        'feature7': 'Religiosity',
        'feature8': 'Education',
        'feature9': 'Occupation',
        'feature10': 'MarriageHappiness'
    }
    df = df.rename(columns=rename_map)

    # Convert AffairCount to numeric (coerce errors) - these are coded counts/top-coded values
    df['AffairCount'] = pd.to_numeric(df['AffairCount'], errors='coerce')

    # Binary indicator for any affair (0 = none, 1 = at least one)
    df['AffairBinary'] = (df['AffairCount'] > 0).astype(int)

    # Map HasChildren to 0/1
    df['HasChildren'] = df['HasChildren_raw'].map({
        'yes': 1,
        'no': 0,
        'Yes': 1,
        'No': 0,
        1: 1,
        0: 0
    })

    # Standardize Gender column and create IsMale indicator.
    # Keep original Gender string for reference but create IsMale numeric control.
    df['Gender'] = df['Gender'].astype(str).str.strip().str.lower()
    df['IsMale'] = df['Gender'].map({'male': 1, 'female': 0})

    # Convert numeric columns to numeric dtype (coerce if bad values)
    numeric_cols = ['Age', 'YearsMarried', 'Religiosity', 'Education', 'Occupation', 'MarriageHappiness']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing values in primary variables (AffairCount or HasChildren)
    df = df.dropna(subset=['AffairCount', 'HasChildren'])

    # For modeling it's useful to drop rows with missing controls (or alternatively impute).
    # Here we drop rows missing key controls to keep models straightforward.
    df = df.dropna(subset=['IsMale', 'Age', 'YearsMarried', 'Religiosity', 'Education', 'Occupation', 'MarriageHappiness'])

    # Reset index
    df = df.reset_index(drop=True)

    # Return only columns required for analysis (plus RespondentID and raw Gender for reference)
    keep_cols = [
        'RespondentID', 'AffairCount', 'AffairBinary', 'HasChildren', 'Gender', 'IsMale',
        'Age', 'YearsMarried', 'Religiosity', 'Education', 'Occupation', 'MarriageHappiness'
    ]
    # If some expected columns are missing, this will raise; that's desirable so user can inspect.
    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit two complementary models to estimate the relationship between having children and engagement in extramarital affairs:
      1) Zero-Inflated Negative Binomial (ZINB) on AffairCount (primary model for count data with excess zeros and overdispersion).
      2) Logistic regression on AffairBinary (secondary model for the extensive margin: any affair vs none).

    Returns a dictionary with fitted model result objects and prints summaries.
    """
    import statsmodels.api as sm
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

    results = {}

    # Define covariates (exogenous variables). Include constant.
    covariate_cols = ['HasChildren', 'IsMale', 'Age', 'YearsMarried', 'Religiosity', 'Education', 'Occupation', 'MarriageHappiness']

    # Prepare exog and endog. Add constant for regression.
    exog = df[covariate_cols].astype(float)
    exog = sm.add_constant(exog, has_constant='add')
    endog_count = df['AffairCount'].astype(float)
    endog_bin = df['AffairBinary'].astype(int)

    # 1) Zero-Inflated Negative Binomial
    try:
        zinb_model = ZeroInflatedNegativeBinomialP(endog_count, exog, exog_infl=exog, inflation='logit')
        zinb_res = zinb_model.fit(method='bfgs', maxiter=200, disp=False)
        print('ZINB model converged:')
        print(zinb_res.summary())
        results['zinb'] = zinb_res
    except Exception as e:
        # If ZINB fails to converge or errors, capture the exception
        print('ZINB model failed:', e)
        results['zinb_error'] = str(e)

    # 2) Logistic regression on any-affair (extensive margin)
    try:
        logit_model = sm.Logit(endog_bin, exog)
        logit_res = logit_model.fit(disp=False)
        print('\nLogistic regression (any affair) results:')
        print(logit_res.summary())
        results['logit'] = logit_res
    except Exception as e:
        print('Logit model failed:', e)
        results['logit_error'] = str(e)

    # Additional simple OLS as robustness check (not ideal for counts, shown for comparison)
    try:
        ols_model = sm.OLS(endog_count, exog)
        ols_res = ols_model.fit(cov_type='HC3')
        print('\nOLS (robust SE) results (for comparison only):')
        print(ols_res.summary())
        results['ols_robust'] = ols_res
    except Exception as e:
        results['ols_error'] = str(e)

    return results


