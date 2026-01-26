from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/replace_with_rvs_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataset for modeling. Transformations performed:
      - Drop rows missing essential variables (y, age, culture).
      - Create mean-centered age (age_centered).
      - Create a categorical site variable (culture_cat) as a string/category.
      - Recode gender to binary is_male (0 = girl, 1 = boy).
      - Ensure majority_first is integer 0/1.
      - Create two derived binary outcomes for follow-up analyses:
          * SociallyGuided: 1 if child chose a demonstrated option (majority or minority), 0 if chose undemonstrated option.
          * Majority_vs_Minority: among socially guided choices, 1 if majority (y==2), 0 if minority (y==3). For y==1 this column will be NaN.
      - Create y_adj = y - 1 (0/1/2) which is convenient for some modeling functions.

    The returned dataframe contains all columns referenced in the modeling code: ['y','y_adj','age_centered','culture_cat','is_male','majority_first','SociallyGuided','Majority_vs_Minority']
    """
    df = df.copy()

    # Ensure required columns exist
    required = ['y', 'age', 'culture', 'gender', 'majority_first']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Drop rows with missing core variables
    df = df.dropna(subset=['y', 'age', 'culture'])

    # Coerce types
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df.dropna(subset=['age'])
    df['culture'] = pd.to_numeric(df['culture'], errors='coerce').astype(int)

    # Mean-center age for interpretability
    df['age_centered'] = df['age'] - df['age'].mean()

    # Create a categorical label for culture (keep numeric id but as category)
    # Use generic site labels (site_1, site_2, ...); if you have a mapping to real site names, replace here.
    df['culture_cat'] = df['culture'].apply(lambda x: f"site_{int(x)}").astype('category')

    # Recode gender to is_male (0 = girl (1), 1 = boy (2))
    # If genders are encoded differently in some rows, non-matching values will become NaN and will be left as-is.
    df['is_male'] = df['gender'].apply(lambda v: 1 if v == 2 else (0 if v == 1 else np.nan)).astype('float')

    # Ensure majority_first is 0/1
    df['majority_first'] = pd.to_numeric(df['majority_first'], errors='coerce').fillna(0).astype(int)

    # Primary dependent variable is y (1,2,3). Create y_adj = y-1 (0,1,2) for some modeling functions
    df['y'] = pd.to_numeric(df['y'], errors='coerce').astype(int)
    df = df[df['y'].isin([1, 2, 3])].copy()
    df['y_adj'] = (df['y'] - 1).astype(int)

    # Derived binary outcomes for focused analyses
    df['SociallyGuided'] = df['y'].isin([2, 3]).astype(int)  # 1 if chose a demonstrated option (majority or minority)

    # Majority vs minority among socially guided choices: 1 = majority (y==2); 0 = minority (y==3); NaN for y==1
    df['Majority_vs_Minority'] = df['y'].map({2: 1, 3: 0})

    # Optionally create coarse age groups (developmental stages) for descriptive tables or plots
    # bins chosen to reflect early childhood, middle childhood, late childhood, early adolescence
    bins = [3.5, 6.5, 9.5, 12.5, 14.5]
    labels = ['4-6', '7-9', '10-12', '13-14']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=True)

    # Final housekeeping: reset index
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Run the statistical models to answer whether reliance on social information and preference for majority vary across cultures and development.

    Analyses performed:
      1) Multinomial logistic regression predicting the three-category choice (y: 1=unchosen, 2=majority, 3=minority)
         from age_centered, culture (as categorical dummies), their interaction, and controls (is_male, majority_first).
      2) Binary logistic regression predicting SociallyGuided (1 if chose a demonstrated option (2 or 3), 0 if undemonstrated (1)).
      3) Among socially-guided trials only, logistic regression predicting Majority_vs_Minority (1=majority, 0=minority).

    Returns a dictionary of fitted statsmodels result objects for each model.
    """
    import patsy
    import statsmodels.api as sm

    results = {}

    # Check required columns
    req = ['y_adj', 'age_centered', 'culture_cat', 'is_male', 'majority_first', 'SociallyGuided', 'Majority_vs_Minority']
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Transformed dataframe missing columns required for modeling: {missing}")

    # Build design matrix for predictors. Use patsy to get consistent dummy coding for culture.
    # We include interaction between age_centered and culture_cat to test whether developmental change differs across sites.
    formula_exog = 'age_centered * C(culture_cat) + is_male + majority_first'

    # 1) Multinomial logistic regression (three-category outcome)
    try:
        exog = patsy.dmatrix(formula_exog, df, return_type='dataframe')
        exog = sm.add_constant(exog, has_constant='add')
        endog = df['y_adj'].astype(int)  # 0,1,2

        mnlogit = sm.MNLogit(endog, exog)
        mnlogit_res = mnlogit.fit(method='newton', maxiter=200, disp=False)
        results['multinomial'] = mnlogit_res
    except Exception as e:
        results['multinomial'] = e

    # 2) Logistic regression for SociallyGuided (demonstrated vs undemonstrated)
    try:
        exog_bin = patsy.dmatrix(formula_exog, df, return_type='dataframe')
        exog_bin = sm.add_constant(exog_bin, has_constant='add')
        y_bin = df['SociallyGuided'].astype(int)
        logit = sm.Logit(y_bin, exog_bin)
        logit_res = logit.fit(disp=False)
        results['social_use_logit'] = logit_res
    except Exception as e:
        results['social_use_logit'] = e

    # 3) Majority vs Minority among socially-guided choices
    df_sg = df[df['SociallyGuided'] == 1].copy()
    if df_sg.shape[0] < 10:
        # Not enough data to fit a reliable model
        results['majority_vs_minority_logit'] = ValueError('Not enough socially-guided observations to fit majority/minority model')
    else:
        try:
            exog_mm = patsy.dmatrix(formula_exog, df_sg, return_type='dataframe')
            exog_mm = sm.add_constant(exog_mm, has_constant='add')
            y_mm = df_sg['Majority_vs_Minority'].astype(int)
            logit_mm = sm.Logit(y_mm, exog_mm)
            logit_mm_res = logit_mm.fit(disp=False)
            results['majority_vs_minority_logit'] = logit_mm_res
        except Exception as e:
            results['majority_vs_minority_logit'] = e

    # Return the fitted result objects (or exceptions) so the caller can inspect summaries.
    return results


