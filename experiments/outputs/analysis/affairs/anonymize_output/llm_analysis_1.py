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
    # Work on a copy
    df = df.copy()

    # Dependent variable: raw frequency as provided (feature2).
    # Values in feature2 are codes: 0 = none, 1 = once, 2 = twice, 3 = 3 times,
    # 7 = 4-10 times (commonly encoded as 7 in this dataset), 12 = monthly/weekly/daily (top-coded as 12).
    df['AffairCount'] = pd.to_numeric(df['feature2'], errors='coerce')
    # Binary indicator: any affair vs none
    df['AnyAffair'] = (df['AffairCount'] > 0).astype(int)

    # Independent variable: children in the marriage (feature6)
    # Normalize case and map to 0/1
    df['feature6_str'] = df['feature6'].astype(str).str.strip().str.lower()
    df['HasChildren'] = df['feature6_str'].map({'yes': 1, 'no': 0})

    # Controls: create clean numeric control variables
    # Gender: map female -> 1, male -> 0 (handles capitalization)
    df['feature3_str'] = df['feature3'].astype(str).str.strip().str.lower()
    df['Female'] = df['feature3_str'].map({'female': 1, 'male': 0})

    # Numeric variables: coerce to numeric, preserving NA for invalid entries
    df['Age'] = pd.to_numeric(df['feature4'], errors='coerce')
    df['YearsMarried'] = pd.to_numeric(df['feature5'], errors='coerce')
    df['Religiosity'] = pd.to_numeric(df['feature7'], errors='coerce')
    df['Education'] = pd.to_numeric(df['feature8'], errors='coerce')
    df['Occupation'] = pd.to_numeric(df['feature9'], errors='coerce')
    df['MaritalHappiness'] = pd.to_numeric(df['feature10'], errors='coerce')

    # Drop helper string columns used for mapping
    df = df.drop(columns=[c for c in ['feature6_str', 'feature3_str'] if c in df.columns])

    # Drop rows missing any variable required for modeling
    required_cols = [
        'AffairCount', 'AnyAffair', 'HasChildren', 'Female', 'Age', 'YearsMarried',
        'Religiosity', 'Education', 'Occupation', 'MaritalHappiness'
    ]
    df = df.dropna(subset=required_cols)

    # Force integer type for binary indicators
    df['AnyAffair'] = df['AnyAffair'].astype(int)
    df['HasChildren'] = df['HasChildren'].astype(int)
    df['Female'] = df['Female'].astype(int)

    # Return the transformed dataframe containing all columns used in modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Runs two complementary specifications to examine whether having children decreases engagement in extramarital affairs:
      1) Logistic regression for the probability of any affair (AnyAffair).
      2) OLS for the intensity (AffairCount) among respondents who reported any affair (>0).

    Returns a dictionary with fitted model objects and printed summaries.
    """
    import statsmodels.api as sm

    results = {}

    # Copy to avoid side effects
    data = df.copy()

    # Predictor matrix (same for both models)
    predictors = ['HasChildren', 'Female', 'Age', 'YearsMarried', 'Religiosity', 'Education', 'Occupation', 'MaritalHappiness']
    X = data[predictors]
    X = sm.add_constant(X)

    # 1) Logistic regression for any affair
    y_bin = data['AnyAffair']
    try:
        logit_mod = sm.Logit(y_bin, X)
        logit_res = logit_mod.fit(disp=False)
        # Robust (HC3) standard errors
        logit_res_robust = logit_res.get_robustcov_results(cov_type='HC3')
        results['logit_model'] = logit_res_robust
        print('Logistic regression (AnyAffair) summary:')
        print(logit_res_robust.summary())
    except Exception as e:
        results['logit_error'] = str(e)
        print('Logit model failed:', e)

    # 2) OLS among respondents who reported any affair (AffairCount > 0)
    data_pos = data[data['AffairCount'] > 0].copy()
    if len(data_pos) >= 30:
        X_pos = sm.add_constant(data_pos[predictors])
        y_pos = data_pos['AffairCount']
        ols_mod = sm.OLS(y_pos, X_pos)
        ols_res = ols_mod.fit(cov_type='HC3')
        results['ols_positive_model'] = ols_res
        print('\nOLS among respondents with Affairs (AffairCount > 0) summary:')
        print(ols_res.summary())
    else:
        # If too few positive cases, fit but warn
        if len(data_pos) > 0:
            X_pos = sm.add_constant(data_pos[predictors])
            y_pos = data_pos['AffairCount']
            try:
                ols_mod = sm.OLS(y_pos, X_pos)
                ols_res = ols_mod.fit(cov_type='HC3')
                results['ols_positive_model'] = ols_res
                print('\nOLS among respondents with Affairs (small N) summary:')
                print(ols_res.summary())
            except Exception as e:
                results['ols_error'] = str(e)
                print('OLS (positive sample) failed:', e)
        else:
            results['ols_positive_model'] = None
            print('\nNo respondents reported an affair; OLS on positive sample not estimated.')

    # Return dictionary of results (fitted model objects or error messages)
    return results


