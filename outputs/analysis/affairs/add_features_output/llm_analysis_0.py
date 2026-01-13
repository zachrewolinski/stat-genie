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
    # Make a copy to avoid modifying original
    df = df.copy()

    # Keep only rows with non-missing outcome and the primary IV
    required_cols = ['affairs', 'children', 'gender', 'age', 'yearsmarried', 'education', 'religiousness', 'rating', 'occupation']
    df = df.dropna(subset=required_cols)

    # Ensure numeric types for affairs
    df['affairs'] = pd.to_numeric(df['affairs'], errors='coerce')

    # Encode children: expected values 'yes'/'no' (case-insensitive). Create HasChildren binary column.
    df['children'] = df['children'].astype(str).str.strip().str.lower()
    df['HasChildren'] = df['children'].map({'yes': 1, 'no': 0})
    # If mapping produced NaN (unexpected values), fallback: treat any non-'no' as 1 when non-empty
    mask_unknown_children = df['HasChildren'].isna() & df['children'].notna()
    if mask_unknown_children.any():
        df.loc[mask_unknown_children, 'HasChildren'] = df.loc[mask_unknown_children, 'children'].apply(lambda x: 0 if x == 'no' else 1)
    df['HasChildren'] = df['HasChildren'].astype(int)

    # Encode gender: create IsFemale (1 = female, 0 = male). Accept common string forms.
    df['gender'] = df['gender'].astype(str).str.strip().str.lower()
    df['IsFemale'] = df['gender'].map({'female': 1, 'male': 0})
    # If missing mapping, try to infer from single-letter coding
    mask_gender_na = df['IsFemale'].isna() & df['gender'].notna()
    if mask_gender_na.any():
        df.loc[mask_gender_na, 'IsFemale'] = df.loc[mask_gender_na, 'gender'].apply(lambda x: 1 if x.startswith('f') else (0 if x.startswith('m') else np.nan))
    df['IsFemale'] = df['IsFemale'].astype(float)

    # Convert numeric covariates to numeric types
    for col in ['age', 'yearsmarried', 'education', 'religiousness', 'rating', 'occupation']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with any remaining missing values in modelling columns
    model_cols = ['affairs', 'HasChildren', 'IsFemale', 'age', 'yearsmarried', 'education', 'religiousness', 'rating', 'occupation']
    df = df.dropna(subset=model_cols)

    # Standardize continuous controls for easier interpretation in the regression
    # Create standardized columns with suffix _s
    for col in ['age', 'yearsmarried', 'education', 'religiousness', 'rating', 'occupation']:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        # Avoid division by zero
        if std == 0 or np.isnan(std):
            df[f'{col}_s'] = 0.0
        else:
            df[f'{col}_s'] = (df[col] - mean) / std

    # Keep only columns required for the model (but do not remove original raw columns)
    final_cols = ['affairs', 'HasChildren', 'IsFemale', 'age_s', 'yearsmarried_s', 'education_s', 'religiousness_s', 'rating_s', 'occupation_s']
    # Ensure final columns exist (they should); select and return dataframe with these plus originals
    # Return full df (with new columns). Caller/model will use these specific columns.
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    # Import model class
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
    import statsmodels.api as sm

    # Ensure required columns are present
    required = ['affairs', 'HasChildren', 'IsFemale', 'age_s', 'yearsmarried_s', 'education_s', 'religiousness_s', 'rating_s', 'occupation_s']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Prepare endogenous and exogenous matrices
    endog = df['affairs'].astype(float)

    exog_vars = ['HasChildren', 'IsFemale', 'age_s', 'yearsmarried_s', 'education_s', 'religiousness_s', 'rating_s', 'occupation_s']
    exog = sm.add_constant(df[exog_vars], has_constant='add')

    # For the zero-inflation (logit) part, use a subset of predictors; here use the same covariates excluding occupation (to keep inflation model more parsimonious)
    exog_infl_vars = ['HasChildren', 'IsFemale', 'age_s', 'yearsmarried_s', 'education_s', 'religiousness_s', 'rating_s']
    exog_infl = sm.add_constant(df[exog_infl_vars], has_constant='add')

    # Fit a Zero-Inflated Negative Binomial model to account for excess zeros and overdispersion
    zinb = ZeroInflatedNegativeBinomialP(endog, exog, exog_infl=exog_infl, inflation='logit')

    # Fit model with default start params; increase maxiter for convergence if needed
    try:
        results = zinb.fit(method='bfgs', maxiter=1000, disp=False)
    except Exception:
        # fallback to default optimization if BFGS fails
        results = zinb.fit(disp=False)

    # Compute and display a compact summary and the marginal effect of HasChildren on expected count
    # (average marginal effect estimated numerically)
    try:
        summary_text = results.summary().as_text()
    except Exception:
        summary_text = str(results.summary())

    # Average marginal effect for HasChildren on expected number of affairs
    # We'll compute predicted expected counts at HasChildren=1 and HasChildren=0 holding other covariates at their observed values,
    # then take the mean difference (average treatment effect)
    df_model = exog.copy()
    df_model_zero = df_model.copy()
    df_model_one = df_model.copy()
    df_model_zero['HasChildren'] = 0
    df_model_one['HasChildren'] = 1

    # Predictions: use results.predict with exog and exog_infl provided
    try:
        mu_zero = results.predict(exog=df_model_zero, exog_infl=exog_infl.loc[df_model_zero.index], which='mean')
        mu_one = results.predict(exog=df_model_one, exog_infl=exog_infl.loc[df_model_one.index], which='mean')
        avg_effect = (mu_one - mu_zero).mean()
    except Exception:
        # If the predict signature differs, fallback to using exog only
        try:
            mu_zero = results.predict(df_model_zero)
            mu_one = results.predict(df_model_one)
            avg_effect = (mu_one - mu_zero).mean()
        except Exception:
            avg_effect = np.nan

    # Attach an interpretation attribute to results for convenience
    results.ame_HasChildren = avg_effect
    results.model_summary_text = summary_text

    # Return the fitted results object (with added attributes ame_HasChildren and model_summary_text)
    return results


