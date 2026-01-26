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

    # Ensure required columns exist and drop rows with missing key values
    required = ['y', 'age', 'culture', 'gender', 'majority_first']
    df = df.dropna(subset=required)

    # Make sure data types are numeric where expected
    df['y'] = pd.to_numeric(df['y'], errors='coerce').astype('Int64')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['culture'] = df['culture'].astype('category')
    df['gender'] = pd.to_numeric(df['gender'], errors='coerce')
    df['majority_first'] = pd.to_numeric(df['majority_first'], errors='coerce').fillna(0).astype(int)

    # Derived dependent variables for secondary analyses
    # SocialReliance: chose one of the demonstrated options (majority or minority) vs the undemonstrated third option
    df['SocialReliance'] = df['y'].apply(lambda v: 1 if v in [2, 3] else 0).astype(int)
    # MajorityChoice: chose the majority option (y == 2)
    df['MajorityChoice'] = df['y'].apply(lambda v: 1 if v == 2 else 0).astype(int)

    # Center age for interpretability
    df['age_c'] = df['age'] - df['age'].mean()

    # Coarse developmental stage bins (useful for stage-wise moderation/summary)
    bins = [3, 6, 9, 12, 14]
    labels = ['4-6', '7-9', '10-12', '13-14']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=True)

    # Categorical culture variable (string/category) for use in model design matrices
    df['culture_cat'] = df['culture'].astype('category')

    # Gender indicator: is_boy (1 if gender==2 per data dictionary, 0 if gender==1)
    df['is_boy'] = df['gender'].apply(lambda g: 1 if g == 2 else 0).astype(int)

    # Keep only rows with valid age_group (should be children 4-14 per dataset, but be safe)
    df = df[~df['age_group'].isna()]

    # Reindex after filtering
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    import statsmodels.api as sm
    import numpy as np
    import pandas as pd

    # Create a working copy
    df = df.copy()

    # ========== Multinomial model: choice (1/2/3) as function of age, culture, and controls ==========
    # Prepare culture dummies (drop first to avoid multicollinearity)
    culture_dummies = pd.get_dummies(df['culture_cat'], prefix='culture', drop_first=True)

    # Base exogenous variables: age (centered), gender, order effect
    exog_base = pd.concat([df[['age_c', 'is_boy', 'majority_first']].reset_index(drop=True), culture_dummies.reset_index(drop=True)], axis=1)

    # Add interactions between age and each culture dummy to test whether age-developmental trajectories differ across cultures
    # (This explicitly tests whether developmental effects vary by culture.)
    for col in culture_dummies.columns:
        exog_base[f'{col}_x_age'] = exog_base[col] * exog_base['age_c']

    exog_base = sm.add_constant(exog_base, has_constant='add')

    # Endogenous variable for multinomial: y (must be numeric array)
    y_multi = df['y'].astype(int)

    # Fit the multinomial logistic regression
    # Use method='newton' / default; suppress iteration output
    try:
        mnlogit = sm.MNLogit(y_multi, exog_base)
        mnlogit_res = mnlogit.fit(disp=False, maxiter=200)
    except Exception as e:
        mnlogit_res = e

    # ========== Logistic model 1: SocialReliance (binary) ==========
    y_sr = df['SocialReliance'].astype(int)

    # Use the same exogenous variables (constant + base + culture dummies + interactions)
    exog_sr = exog_base.copy()
    try:
        logit_sr = sm.Logit(y_sr, exog_sr)
        logit_sr_res = logit_sr.fit(disp=False, maxiter=200)
    except Exception as e:
        logit_sr_res = e

    # ========== Logistic model 2: Majority preference among children who used social information ==========
    df_social = df[df['SocialReliance'] == 1].copy()
    if df_social.shape[0] >= 20:
        # Recreate culture dummies for subset (drop_first=True) and exog
        culture_dummies_sub = pd.get_dummies(df_social['culture_cat'], prefix='culture', drop_first=True)
        exog_mp = pd.concat([df_social[['age_c', 'is_boy', 'majority_first']].reset_index(drop=True), culture_dummies_sub.reset_index(drop=True)], axis=1)
        # interactions age x culture for subset
        for col in culture_dummies_sub.columns:
            exog_mp[f'{col}_x_age'] = exog_mp[col] * exog_mp['age_c']
        exog_mp = sm.add_constant(exog_mp, has_constant='add')

        y_mp = df_social['MajorityChoice'].astype(int)
        try:
            logit_mp = sm.Logit(y_mp, exog_mp)
            logit_mp_res = logit_mp.fit(disp=False, maxiter=200)
        except Exception as e:
            logit_mp_res = e
    else:
        logit_mp_res = ValueError('Not enough cases with SocialReliance==1 to fit majority-preference model (need >= 20 rows)')

    # Package results. Returning fitted result objects (or exceptions) so caller can inspect summaries.
    results = {
        'multinomial_result': mnlogit_res,
        'social_reliance_logit_result': logit_sr_res,
        'majority_pref_logit_result': logit_mp_res,
        # Also return some model-ready DataFrame slices for further inspection if needed
        'exog_multinomial_columns': list(exog_base.columns),
        'n_rows': df.shape[0],
        'n_social_rows': int(df['SocialReliance'].sum())
    }

    return results


