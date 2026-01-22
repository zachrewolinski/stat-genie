from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/replace_and_positive_statement_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for multinomial modeling.

    Produces the following new columns used by the model:
      - y_cat: outcome encoded as 0 (unchosen), 1 (majority), 2 (minority)
      - is_boy: binary gender indicator (1=boy, 0=girl)
      - age_center: centered age (age - mean(age))
      - C_2..C_8: culture dummy columns (C_1 used as reference and dropped)
      - age_x_C_*: interaction terms between age_center and each culture dummy
      - majority_first: ensured as integer 0/1

    Rows with missing critical fields (y, age, culture, gender, majority_first) are dropped.
    """
    df = df.copy()

    # Drop rows missing essential variables
    df = df.dropna(subset=['y', 'age', 'culture', 'gender', 'majority_first'])

    # Ensure integer types where appropriate
    df['y'] = df['y'].astype(int)
    df['age'] = df['age'].astype(float)
    df['culture'] = df['culture'].astype(int)
    df['gender'] = df['gender'].astype(int)
    df['majority_first'] = df['majority_first'].astype(int)

    # Dependent variable: map to 0,1,2 for multinomial modeling
    # Original coding: 1=unchosen, 2=majority, 3=minority
    df['y_cat'] = (df['y'] - 1).astype(int)  # now 0,1,2

    # Controls
    df['is_boy'] = (df['gender'] == 2).astype(int)

    # Center age to improve interpretability and reduce collinearity with interactions
    df['age_center'] = df['age'] - df['age'].mean()

    # Culture dummies: create one-hot columns for cultures, drop first (culture 1 as reference)
    # Expecting culture IDs 1..8 per schema; this will create C_2..C_8
    culture_dummies = pd.get_dummies(df['culture'].astype(int), prefix='C', drop_first=True)
    df = pd.concat([df, culture_dummies], axis=1)

    # Create interaction terms between centered age and each culture dummy
    culture_cols = [c for c in df.columns if c.startswith('C_')]
    for c in culture_cols:
        df[f'age_x_{c}'] = df['age_center'] * df[c]

    # Create additional binary outcomes for exploratory checks (not the primary DV but useful later):
    # relied_on_social: chose a demonstrated option (majority or minority) vs unchosen
    df['relied_on_social'] = (df['y_cat'] != 0).astype(int)
    # majority_choice: chose majority vs not
    df['majority_choice'] = (df['y_cat'] == 1).astype(int)

    # Final check: ensure all model columns exist. If a culture level was absent in the sample,
    # the corresponding C_* column will not exist; downstream model code locates culture and interaction
    # columns dynamically, so this is acceptable.

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a multinomial logistic regression to test whether children's choices (unchosen / majority / minority)
    vary with age and across cultures, and whether developmental trajectories (age effects) differ by culture.

    The primary model is a full multinomial logistic model with predictors:
      - age_center
      - is_boy
      - majority_first
      - culture dummies C_2..C_8 (C_1 implicit reference)
      - interactions age_center x culture_dummies

    We fit a reduced model without the age x culture interactions and perform a likelihood-ratio test
    to evaluate whether developmental trajectories differ across cultures (i.e., whether age effects vary by culture).

    Returns a dictionary with full and reduced model results and LR-test statistics.
    """
    import statsmodels.api as sm
    import scipy.stats as stats

    results = {}

    # Prepare endogenous (y) and exogenous (X) matrices. Locate culture & interaction columns dynamically.
    endog = df['y_cat'].astype(int)

    # Identify culture dummy columns and interaction columns produced in transform()
    culture_cols = sorted([c for c in df.columns if c.startswith('C_')])  # e.g., C_2..C_8
    interaction_cols = sorted([c for c in df.columns if c.startswith('age_x_C_')])

    # Base controls and main predictors
    base_cols = ['age_center', 'is_boy', 'majority_first']

    # Build exogenous (full) matrix with constant
    exog_cols_full = base_cols + culture_cols + interaction_cols
    # If any expected columns are missing (e.g., a culture not present), filter them out gracefully
    exog_cols_full = [c for c in exog_cols_full if c in df.columns]
    exog_full = sm.add_constant(df[exog_cols_full], has_constant='add')

    # Fit full multinomial logit
    full_model = sm.MNLogit(endog, exog_full)
    full_res = full_model.fit(disp=False, maxiter=200)

    # Reduced model: drop interaction terms (tests whether age-by-culture interactions improve fit)
    exog_cols_reduced = [c for c in exog_cols_full if not c.startswith('age_x_')]
    exog_reduced = sm.add_constant(df[exog_cols_reduced], has_constant='add')
    reduced_model = sm.MNLogit(endog, exog_reduced)
    reduced_res = reduced_model.fit(disp=False, maxiter=200)

    # Likelihood-ratio test for interactions
    llf_full = full_res.llf
    llf_reduced = reduced_res.llf
    lr_stat = 2.0 * (llf_full - llf_reduced)
    df_diff = int(full_res.df_model - reduced_res.df_model)
    lr_pvalue = stats.chi2.sf(lr_stat, df_diff) if df_diff > 0 else float('nan')

    results['full_result'] = full_res
    results['reduced_result'] = reduced_res
    results['lr_test_interactions'] = {
        'lr_stat': float(lr_stat),
        'df_diff': df_diff,
        'p_value': float(lr_pvalue)
    }

    # Additional complementary tests / summaries for interpretation
    # 1) Test whether culture main effects (differences in intercepts) are significant by comparing
    #    the reduced model above to a model that also drops culture main effects.
    exog_cols_no_culture = [c for c in exog_cols_reduced if not c.startswith('C_')]
    exog_no_culture = sm.add_constant(df[exog_cols_no_culture], has_constant='add')
    model_no_culture = sm.MNLogit(endog, exog_no_culture)
    res_no_culture = model_no_culture.fit(disp=False, maxiter=200)

    llf_reduced2 = res_no_culture.llf
    lr_stat_culture = 2.0 * (reduced_res.llf - llf_reduced2)
    df_diff_culture = int(reduced_res.df_model - res_no_culture.df_model)
    pval_culture = stats.chi2.sf(lr_stat_culture, df_diff_culture) if df_diff_culture > 0 else float('nan')

    results['test_culture_main_effects'] = {
        'lr_stat': float(lr_stat_culture),
        'df_diff': df_diff_culture,
        'p_value': float(pval_culture)
    }

    # 2) Fit binary logistic models for complementary, easy-to-interpret checks:
    #    (a) relied_on_social (demonstrated option vs unchosen)
    #    (b) majority_choice (majority vs others)
    # These use sklearn's logistic regression (or statsmodels) for quick checks.
    try:
        from sklearn.linear_model import LogisticRegression
        X_binary = df[base_cols + culture_cols]
        X_binary = pd.get_dummies(X_binary, drop_first=False)  # safe fallback; will include all numerics as-is
        # fill na if any
        X_binary = X_binary.fillna(0)

        # (a) relied_on_social
        y_relied = df['relied_on_social'].astype(int)
        clf_relied = LogisticRegression(penalty='none', solver='saga', max_iter=10000)
        clf_relied.fit(X_binary, y_relied)

        # (b) majority_choice
        y_maj = df['majority_choice'].astype(int)
        clf_maj = LogisticRegression(penalty='none', solver='saga', max_iter=10000)
        clf_maj.fit(X_binary, y_maj)

        results['binary_checks'] = {
            'relied_on_social_coef': dict(zip(X_binary.columns.tolist(), clf_relied.coef_[0].tolist())),
            'majority_choice_coef': dict(zip(X_binary.columns.tolist(), clf_maj.coef_[0].tolist()))
        }
    except Exception:
        # If sklearn isn't available or numeric issues occur, skip binary checks
        results['binary_checks'] = None

    # Attach brief human-readable summaries
    results['summary'] = {
        'n_obs': int(df.shape[0]),
        'n_culture_levels_observed': len(culture_cols) + 1  # +1 for the dropped reference level C_1
    }

    return results


