from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/positive_leading_statement_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset into analysis-ready DataFrame with:
      - outcomes: y (kept), social_choice (binary), majority_choice (binary among demonstrated choices)
      - predictors / controls: age_centered, age_centered_sq, age_group, gender_female, majority_first
      - culture dummy variables: culture_2..culture_8 (drop_first=True)
      - age-by-culture interaction columns named age_x_<culture_dummy>

    The function returns a copy of df with added columns. Rows with missing critical values (y, age, culture) are dropped.
    """
    df = df.copy()

    # Ensure key columns exist
    for col in ['y', 'age', 'culture', 'gender', 'majority_first']:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")

    # Cast to numeric where appropriate (coerce non-numeric to NaN), then drop rows missing critical variables
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['culture'] = pd.to_numeric(df['culture'], errors='coerce')
    df['gender'] = pd.to_numeric(df['gender'], errors='coerce')
    df['majority_first'] = pd.to_numeric(df['majority_first'], errors='coerce')

    # Drop rows missing critical variables
    df = df.dropna(subset=['y', 'age', 'culture']).copy()

    # Now safely convert to integer types where appropriate (these should have no NaNs now)
    df['y'] = df['y'].astype(int)
    df['culture'] = df['culture'].astype(int)
    # gender and majority_first may still contain NaNs for some rows; keep them as nullable integers
    df['gender'] = pd.to_numeric(df['gender'], errors='coerce').astype(pd.Int64Dtype())
    df['majority_first'] = pd.to_numeric(df['majority_first'], errors='coerce').astype(pd.Int64Dtype())

    # Dependent variables
    # raw categorical outcome is kept as 'y' (1=undemonstrated, 2=majority, 3=minority)
    df['social_choice'] = df['y'].isin([2, 3]).astype(int)  # 1 if child chose a demonstrated option (majority or minority)

    # majority_choice among those who selected a demonstrated option: 1 if majority (y==2), 0 if minority (y==3), NaN otherwise
    df['majority_choice'] = df['y'].map({2: 1, 3: 0})

    # Age-related predictors: center age and add quadratic term to allow nonlinearity
    df['age_centered'] = df['age'] - df['age'].mean()
    df['age_centered_sq'] = df['age_centered'] ** 2

    # Age groups for descriptive purposes
    df['age_group'] = pd.cut(df['age'], bins=[3.5, 6.5, 9.5, 14.5], labels=['4-6', '7-9', '10-14'])

    # Gender control: make female = 1 (original coding: 1 = girl, 2 = boy)
    # If gender is missing, gender_female will be 0 for now; keep original 'gender' column as required
    df['gender_female'] = (df['gender'] == 1).astype(int)

    # Create culture dummy variables for cultures 1..8; drop_first=True to avoid collinearity with intercept
    # We cast culture to string so dummy names are consistent even if some integer levels are missing in a small subset
    culture_dummies = pd.get_dummies(df['culture'].astype(int).astype(str), prefix='culture', drop_first=True)
    # Ensure consistent set of columns for cultures 2..8 even if some are absent in df
    expected_cult_cols = [f'culture_{i}' for i in range(2, 9)]
    for c in expected_cult_cols:
        if c not in culture_dummies.columns:
            culture_dummies[c] = 0
    # Keep only expected order
    culture_dummies = culture_dummies[expected_cult_cols]

    df = pd.concat([df, culture_dummies], axis=1)

    # Interaction terms: age_centered x each culture dummy
    for c in expected_cult_cols:
        interaction_col = 'age_x_' + c
        df[interaction_col] = df['age_centered'] * df[c]

    # Final housekeeping: ensure integer columns are ints and floats are floats
    int_cols = ['y', 'social_choice', 'gender', 'gender_female', 'majority_first']
    for c in int_cols:
        if c in df.columns:
            # Use nullable integer dtype to be robust to any remaining missing values
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype(pd.Int64Dtype())
            except Exception:
                # fallback: leave as-is if conversion fails
                pass

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """\n    Run three complementary analyses to test whether reliance on social information and preference for majority cues vary across cultures and development:\n      1) Multinomial logistic regression predicting raw choice 'y' (1=undemonstrated, 2=majority, 3=minority) from age, culture, gender, majority_first, and culture dummies.\n      2) Binary logistic regression (GLM binomial) predicting 'social_choice' (chose demonstrated vs undemonstrated) with the same predictors plus age^2 to capture nonlinear age effects and interactions.\n      3) Binary logistic regression among children who chose a demonstrated option (y in {2,3}) predicting 'majority_choice' (majority vs minority) to test majority preference across age and cultures.\n\n    Returns a dictionary with fitted model results and ASCII summaries for quick inspection.\n    """
    results = {}

    # Prepare base covariates
    base_covars = ['age_centered', 'age_centered_sq', 'gender_female', 'majority_first']
    culture_cols = [c for c in df.columns if c.startswith('culture_')]
    interaction_cols = [c for c in df.columns if c.startswith('age_x_culture_')]

    # Build design matrix for fixed-effects predictors
    X_cols = base_covars + culture_cols + interaction_cols

    # Ensure no missing columns (if some were entirely absent earlier)
    for c in X_cols:
        if c not in df.columns:
            df[c] = 0

    # 1) Multinomial logistic regression on raw y
    try:
        # MNLogit expects dependent variable coded from 0..(k-1), so shift y down by 1
        y_mn = (df['y'] - 1).astype(int)
        X_mn = df[X_cols].astype(float)
        X_mn = sm.add_constant(X_mn, has_constant='add')

        mn_model = sm.MNLogit(y_mn, X_mn)
        mn_res = mn_model.fit(method='newton', maxiter=200, disp=False)
        results['multinomial_model'] = mn_res
        results['multinomial_summary'] = mn_res.summary().as_text()
    except Exception as e:
        results['multinomial_error'] = str(e)

    # 2) Logistic regression predicting whether child relied on social information (chose demonstrated option)
    try:
        df_soc = df.dropna(subset=['social_choice'])
        y_soc = df_soc['social_choice'].astype(float)
        X_soc = df_soc[X_cols].astype(float)
        # Add constant
        X_soc = sm.add_constant(X_soc, has_constant='add')
        soc_model = sm.GLM(y_soc, X_soc, family=sm.families.Binomial())
        soc_res = soc_model.fit(maxiter=100)
        results['social_choice_model'] = soc_res
        results['social_choice_summary'] = soc_res.summary().as_text()
    except Exception as e:
        results['social_choice_error'] = str(e)

    # 3) Logistic regression among children who chose a demonstrated option: majority vs minority
    try:
        df_demo = df[df['y'].isin([2, 3])].copy()
        if df_demo.shape[0] == 0:
            raise ValueError('No observations where y is 2 or 3; cannot fit majority preference model')
        # majority_choice is 1 for majority (y==2), 0 for minority (y==3)
        df_demo = df_demo.dropna(subset=['majority_choice'])
        y_maj = df_demo['majority_choice'].astype(float)
        X_maj = df_demo[X_cols].astype(float)
        X_maj = sm.add_constant(X_maj, has_constant='add')
        maj_model = sm.GLM(y_maj, X_maj, family=sm.families.Binomial())
        maj_res = maj_model.fit(maxiter=100)
        results['majority_pref_model'] = maj_res
        results['majority_pref_summary'] = maj_res.summary().as_text()
    except Exception as e:
        results['majority_pref_error'] = str(e)

    # Return results (fitted model objects + printable summaries / errors)
    return results