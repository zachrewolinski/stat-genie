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
    # Make a copy
    df = df.copy()

    # Drop rows with missing essential variables
    df = df.dropna(subset=['y', 'age', 'culture'])

    # Ensure types
    df['y'] = df['y'].astype(int)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['culture'] = df['culture'].astype(int)

    # Primary dependent variable: 3-category outcome mapped to 0,1,2 for MNLogit
    # Original mapping: 1=unchosen option, 2=majority option, 3=minority option
    df['y_mn'] = (df['y'] - 1).astype(int)  # now 0=unchosen,1=majority,2=minority

    # Derived binary dependent variables
    # Reliance on social information: chose a demonstrated option (majority or minority) vs unchosen
    df['social_choice'] = df['y'].apply(lambda v: 1 if v in [2, 3] else 0).astype(int)

    # Preference for majority among those who used social information (2 or 3)
    df['majority_among_dem'] = np.where(df['y'].isin([2, 3]), (df['y'] == 2).astype(int), np.nan)

    # Center age to aid interpretation and interaction stability
    df['age_c'] = df['age'] - df['age'].mean()

    # Gender binary control: create gender_male (1 if original gender==2 (boy), 0 if girl (1))
    df['gender_male'] = df['gender'].apply(lambda x: 1 if x == 2 else 0).astype(int)

    # Ensure majority_first is binary (0/1)
    df['majority_first'] = df['majority_first'].astype(int)

    # Create culture dummies (drop first as reference). Use string conversion so dummy names are stable.
    culture_dummies = pd.get_dummies(df['culture'].astype(int).astype(str), prefix='culture', drop_first=True)
    # Rename dummy columns to match expected pattern (e.g., culture_2 ...) - they will already be like 'culture_2'
    # Concatenate dummies
    df = pd.concat([df, culture_dummies], axis=1)

    # Record which culture dummy columns were created for later use
    df['_culture_dummy_cols'] = ','.join(culture_dummies.columns)

    # Create interactions between centered age and each culture dummy (age-by-culture slope differences)
    interaction_cols = []
    for col in culture_dummies.columns:
        inter_name = f'age_c*{col}'
        df[inter_name] = df['age_c'] * df[col]
        interaction_cols.append(inter_name)

    # For convenience, expose the names of interaction columns (useful in modeling code)
    df['_culture_interaction_cols'] = ','.join(interaction_cols)

    # Final dataset keeps original columns plus derived columns. Drop helper columns if you prefer,
    # but keep _culture_dummy_cols and _culture_interaction_cols to programmatically build models later.
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    # Build predictor list programmatically using the helper fields created in transform
    # Base predictors
    base_vars = ['age_c', 'gender_male', 'majority_first']

    # Extract culture dummy columns created in transform (if any)
    culture_dummy_cols = []
    if '_culture_dummy_cols' in df.columns:
        s = df['_culture_dummy_cols'].dropna().unique()
        if len(s) > 0:
            # take first non-empty string
            first = s[0]
            if first != '':
                culture_dummy_cols = first.split(',')

    # Extract interaction columns
    interaction_cols = []
    if '_culture_interaction_cols' in df.columns:
        s2 = df['_culture_interaction_cols'].dropna().unique()
        if len(s2) > 0:
            first2 = s2[0]
            if first2 != '':
                interaction_cols = first2.split(',')

    # Final exogenous variable list
    exog_vars = base_vars + culture_dummy_cols + interaction_cols

    # Guard: ensure the variables actually exist in df
    exog_vars = [v for v in exog_vars if v in df.columns]

    # Add constant
    X = df[exog_vars].copy()
    X = sm.add_constant(X, has_constant='add')

    results = {}

    # 1) Multinomial logistic regression for the 3-category choice (primary test)
    # Endogenous variable: y_mn (0=unchosen,1=majority,2=minority)
    endog = df['y_mn']

    try:
        mnlogit_mod = sm.MNLogit(endog, X)
        mnlogit_res = mnlogit_mod.fit(method='newton', maxiter=200, full_output=True, disp=False)
        print('\nMultinomial logit results:')
        print(mnlogit_res.summary())
        results['mnlogit'] = mnlogit_res
    except Exception as e:
        print('Multinomial logit failed:', e)
        results['mnlogit_error'] = str(e)

    # 2) Binary logistic regression: reliance on social information (any demonstrated option vs unchosen)
    # Test whether social_choice depends on age, culture and their interaction (developmental and cultural variation)
    endog_social = df['social_choice']
    try:
        logit_social_mod = sm.Logit(endog_social, X)
        logit_social_res = logit_social_mod.fit(disp=False)
        print('\nLogit (social_choice) results:')
        print(logit_social_res.summary())
        results['logit_social'] = logit_social_res
    except Exception as e:
        print('Logit (social_choice) failed:', e)
        results['logit_social_error'] = str(e)

    # 3) Binary logistic regression among those who used social information: majority vs minority
    df_maj = df[df['majority_among_dem'].notna()].copy()
    if df_maj.shape[0] >= 20:
        X_maj = X.loc[df_maj.index]
        endog_maj = df_maj['majority_among_dem']
        try:
            logit_maj_mod = sm.Logit(endog_maj, X_maj)
            logit_maj_res = logit_maj_mod.fit(disp=False)
            print('\nLogit (majority among demonstrated) results:')
            print(logit_maj_res.summary())
            results['logit_majority_among_dem'] = logit_maj_res
        except Exception as e:
            print('Logit (majority among demonstrated) failed:', e)
            results['logit_majority_error'] = str(e)
    else:
        msg = 'Not enough observations with demonstrated choices to fit majority-among-dem model.'
        print(msg)
        results['logit_majority_among_dem_error'] = msg

    # The critical tests to answer the research question are:
    # - Main effect of culture dummy variables on choice probabilities (do sites differ?), and
    # - Interaction terms age_c * culture_x (do age-related trajectories differ across cultures?).
    # If culture main effects and/or the age-by-culture interactions are statistically indistinguishable from zero,
    # that supports the hypothesis that reliance on social information and majority preference do NOT vary across cultures and developmental stages.

    return results


