from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/negative_leading_statement_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Drop rows with missing critical values
    df = df.dropna(subset=['y', 'age', 'culture'])

    # Ensure integer types where appropriate
    df['y'] = df['y'].astype(int)
    df['age'] = df['age'].astype(float)
    df['culture'] = df['culture'].astype(int)

    # Primary DV for multinomial models: map 1,2,3 -> 0,1,2 (required by statsmodels MNLogit)
    df['y_code'] = (df['y'] - 1).astype(int)

    # Human-readable category label (not required by models but useful)
    df['y_cat'] = df['y'].map({1: 'unchosen', 2: 'majority', 3: 'minority'})

    # Binary: did the child choose a demonstrated option (majority or minority)?
    df['demonstrated_choice'] = df['y'].isin([2, 3]).astype(int)

    # Among demonstrated choices, did the child pick the majority? (NaN when y==1)
    df['majority_choice'] = np.where(df['y'] == 2, 1, np.where(df['y'] == 3, 0, np.nan)).astype(float)

    # Gender as binary control: 0 = girl (gender==1), 1 = boy (gender==2)
    df['gender_binary'] = df['gender'].map({1: 0, 2: 1}).astype(int)

    # Center age (continuous independent variable)
    df['age_centered'] = df['age'] - df['age'].mean()

    # Derive developmental stages as categorical bins
    # 4-6: early childhood, 7-9: middle childhood, 10-12: late childhood, 13-14: early adolescence
    bins = [3.5, 6.5, 9.5, 12.5, 14.5]
    labels = ['4-6', '7-9', '10-12', '13-14']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, ordered=True)

    # Ensure culture is categorical (string form makes get_dummies explicit later)
    df['culture'] = df['culture'].astype(int).astype(str)

    # Keep useful columns only (but return full dataframe with these new columns)
    # Note: we intentionally do not drop rows with majority_choice == NaN because that variable
    # is only meaningful for the subset of demonstrated choices; downstream models will subset as needed.
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Runs a set of models to test whether children's reliance on social information and
    preference for the majority vary with age and across cultures.

    Returns a dictionary with fitted model objects and a likelihood-ratio test for the joint
    effect of age and culture on the 3-way choice.
    """
    from scipy import stats

    results = {}

    # Prepare exogenous matrix for models: continuous covariates + culture dummies
    base_exog = df[['age_centered', 'gender_binary', 'majority_first']].copy()

    # Create culture dummy variables (drop first to avoid multicollinearity)
    culture_dummies = pd.get_dummies(df['culture'], prefix='culture', drop_first=True)

    exog = pd.concat([base_exog, culture_dummies], axis=1)
    exog = sm.add_constant(exog, has_constant='add')

    # Endogenous variable for multinomial (0,1,2)
    endog = df['y_code'].astype(int)

    # Fit multinomial logistic regression: 3-choice outcome
    try:
        mnlogit_full = sm.MNLogit(endog, exog).fit(method='newton', maxiter=200, disp=False)
        results['mnlogit_full'] = mnlogit_full
    except Exception as e:
        results['mnlogit_full_error'] = str(e)
        mnlogit_full = None

    # Fit reduced multinomial model without age and culture to test joint contribution
    # Reduced exog keeps intercept, gender, majority_first only
    exog_reduced = sm.add_constant(base_exog[['gender_binary', 'majority_first']], has_constant='add')
    try:
        mnlogit_reduced = sm.MNLogit(endog, exog_reduced).fit(method='newton', maxiter=200, disp=False)
        results['mnlogit_reduced'] = mnlogit_reduced
    except Exception as e:
        results['mnlogit_reduced_error'] = str(e)
        mnlogit_reduced = None

    # If both models fit, compute likelihood-ratio test for joint effect of age + culture
    if (mnlogit_full is not None) and (mnlogit_reduced is not None):
        llf_full = mnlogit_full.llf
        llf_reduced = mnlogit_reduced.llf
        lr_stat = 2 * (llf_full - llf_reduced)
        # Degrees of freedom: difference in number of parameters (df_model or params)
        df_full = mnlogit_full.df_model  # total number of regressors across equations
        df_reduced = mnlogit_reduced.df_model
        df_diff = int(df_full - df_reduced)
        p_value = stats.chi2.sf(lr_stat, df_diff) if df_diff > 0 else np.nan
        results['mnlogit_lr_test'] = {
            'lr_stat': float(lr_stat),
            'df_diff': int(df_diff),
            'p_value': float(p_value)
        }
    else:
        results['mnlogit_lr_test'] = 'Could not compute LR test because one or both models failed to fit.'

    # Logistic regression: did child choose any demonstrated option? (binary outcome)
    # This tests general reliance on social information
    try:
        endog_demo = df['demonstrated_choice'].astype(float)
        exog_demo = exog.copy()  # same covariates (age + culture + controls)
        logit_demo = sm.Logit(endog_demo, exog_demo).fit(disp=False, maxiter=200)
        results['logit_demonstrated'] = logit_demo
    except Exception as e:
        results['logit_demonstrated_error'] = str(e)

    # Logistic regression among trials where child chose a demonstrated option: majority vs minority
    df_demo = df[df['demonstrated_choice'] == 1].copy()
    if df_demo.shape[0] >= 20:
        try:
            endog_major = df_demo['majority_choice'].astype(float)
            # Recreate exog for this subset (must align rows)
            exog_major = pd.concat([df_demo[['age_centered', 'gender_binary', 'majority_first']].reset_index(drop=True),
                                     pd.get_dummies(df_demo['culture'], prefix='culture', drop_first=True).reset_index(drop=True)],
                                    axis=1)
            exog_major = sm.add_constant(exog_major, has_constant='add')
            logit_major = sm.Logit(endog_major, exog_major).fit(disp=False, maxiter=200)
            results['logit_majority_given_demonstrated'] = logit_major
        except Exception as e:
            results['logit_majority_given_demonstrated_error'] = str(e)
    else:
        results['logit_majority_given_demonstrated_error'] = 'Too few demonstrated-choice rows to fit model reliably.'

    # Return all relevant objects and the dataframe used (for downstream inspection)
    results['df_used_columns'] = list(df.columns)
    results['n_rows'] = int(df.shape[0])
    return results


