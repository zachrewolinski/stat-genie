from typing import Any, Dict, List, Optional, Set, Tuple, FrozenSet, Literal
import numpy as np
import pandas as pd
import sklearn  # noqa: F401
import scipy  # noqa: F401
import statsmodels.api as sm
import statsmodels.formula.api as smf  # noqa: F401
import matplotlib.pyplot as plt  # noqa: F401
import pickle  # noqa: F401


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy to avoid modifying input in-place
    df = df.copy()

    # Required raw columns: 'y', 'gender', 'age', 'majority_first', 'culture'
    # Drop rows with missing values in core variables
    df = df.dropna(subset=['y', 'age', 'culture'])

    # 1) Basic recodes and derived variables
    # gender: 1 = girl, 2 = boy in dataset. Create is_female indicator (1 = girl, 0 = boy)
    df['is_female'] = (df['gender'] == 1).astype(int)

    # 2) Center age and create coarse age groups (developmental stages)
    df['age_c'] = df['age'] - df['age'].mean()
    # Age groups: 4-6, 7-9, 10-12, 13-14
    bins = [3.5, 6.5, 9.5, 12.5, 14.5]
    labels = ['4-6', '7-9', '10-12', '13-14']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

    # 3) Derived dependent variables
    # social_follow: 1 if child chose a demonstrated option (majority or minority: y==2 or y==3), 0 if chose undemonstrated (y==1)
    df['social_follow'] = df['y'].apply(lambda v: 1 if v in [2, 3] else 0)

    # majority_pref: among those who followed demonstrators, 1 if chose majority (y==2), 0 if chose minority (y==3)
    df['majority_pref'] = pd.NA
    mask_demonstrated = df['y'].isin([2, 3])
    df.loc[mask_demonstrated, 'majority_pref'] = df.loc[mask_demonstrated, 'y'].apply(lambda v: 1 if v == 2 else 0)
    # Convert majority_pref to numeric (0/1) where available; use to_numeric to safely coerce pd.NA to NaN
    df['majority_pref'] = pd.to_numeric(df['majority_pref'], errors='coerce')

    # 4) Culture: keep a categorical version and create one-hot dummies for modeling (drop first to avoid collinearity)
    df['culture_cat'] = df['culture'].astype('category')
    culture_dummies = pd.get_dummies(df['culture_cat'], prefix='culture', drop_first=True)
    # Append dummy columns to df (these will be used as fixed effects and for interactions)
    for col in culture_dummies.columns:
        df[col] = culture_dummies[col]

    # 5) Interaction terms: age_c x each culture dummy to test culture-specific developmental slopes
    # Only use the one-hot dummy columns (e.g., culture_2, culture_3, ...) and exclude 'culture_cat' which is categorical
    culture_dummy_cols = [c for c in df.columns if c.startswith('culture_') and c != 'culture_cat']
    for ccol in culture_dummy_cols:
        inter_name = f'age_c:{ccol}'
        df[inter_name] = df['age_c'] * df[ccol]

    # 6) Ensure majority_first is numeric 0/1 if possible
    df['majority_first'] = pd.to_numeric(df['majority_first'], errors='coerce')
    # If there are no missing values, cast to int to ensure 0/1 integer coding; otherwise leave as numeric with NaN
    if not df['majority_first'].isnull().any():
        df['majority_first'] = df['majority_first'].astype(int)

    # Final check: drop rows with NA in binary derived outcomes when needed for their specific analyses
    # (we keep them here — model function will subset appropriately)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """Run three complementary models to answer the research question:
    1) Multinomial logistic regression predicting the full 3-category choice (y = 1 undemonstrated, 2 majority, 3 minority) from age, culture, their interaction, and controls.
    2) Logistic regression for social_follow (binary: followed demonstrators vs chose undemonstrated) with the same predictors.
    3) Logistic regression for majority_pref among children who followed demonstrators (1 = majority, 0 = minority) with the same predictors.

    Returns a dict with fitted results objects for each model.
    """
    results: Dict[str, Any] = {}

    # Make a copy
    df = df.copy()

    # Identify culture dummy columns created in transform: 'culture_2', 'culture_3', ... depending on site ids
    # Exclude 'culture_cat' which is a categorical column not a dummy
    culture_dummy_cols = [c for c in df.columns if c.startswith('culture_') and c != 'culture_cat']
    interaction_cols = [f'age_c:{c}' for c in culture_dummy_cols]

    # Base predictor set: controls + main effects + culture dummies + interactions
    predictor_cols = ['is_female', 'majority_first', 'age_c'] + culture_dummy_cols + interaction_cols

    # 1) Multinomial logistic regression (full 3-category outcome)
    # Prepare endog as category codes (0..K-1)
    endog_cat = df['y'].astype('category')
    endog = endog_cat.cat.codes

    # Prepare exog and add constant
    exog = df[predictor_cols].astype(float)
    exog = sm.add_constant(exog, has_constant='add')

    # Fit MNLogit. If convergence problems occur, try a different method.
    mnlogit_mod = sm.MNLogit(endog, exog)
    try:
        mnlogit_res = mnlogit_mod.fit(method='newton', maxiter=200, disp=False)
    except Exception:
        mnlogit_res = mnlogit_mod.fit(method='bfgs', maxiter=200, disp=False)

    results['multinomial'] = mnlogit_res

    # 2) Logistic regression for social_follow
    # Drop rows with missing social_follow
    df_sf = df.dropna(subset=['social_follow'])
    endog_sf = df_sf['social_follow'].astype(float)
    exog_sf = df_sf[predictor_cols].astype(float)
    exog_sf = sm.add_constant(exog_sf, has_constant='add')
    logit_sf = sm.Logit(endog_sf, exog_sf)
    try:
        logit_sf_res = logit_sf.fit(disp=False)
    except Exception:
        logit_sf_res = logit_sf.fit(method='bfgs', disp=False)
    results['social_follow_logit'] = logit_sf_res

    # 3) Logistic regression for majority_pref among those who followed demonstrators
    df_mp = df[df['social_follow'] == 1].copy()
    # Drop missing majority_pref if any
    df_mp = df_mp.dropna(subset=['majority_pref'])
    if df_mp.shape[0] >= 20:
        endog_mp = df_mp['majority_pref'].astype(float)
        exog_mp = df_mp[predictor_cols].astype(float)
        exog_mp = sm.add_constant(exog_mp, has_constant='add')
        logit_mp = sm.Logit(endog_mp, exog_mp)
        try:
            logit_mp_res = logit_mp.fit(disp=False)
        except Exception:
            logit_mp_res = logit_mp.fit(method='bfgs', disp=False)
        results['majority_pref_logit'] = logit_mp_res
    else:
        results['majority_pref_logit'] = None

    # Return results dict. Each value is a fitted statsmodels results object (inspect using .summary() externally).
    return results