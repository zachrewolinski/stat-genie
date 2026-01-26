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
    # Ensure required columns exist
    required = ['y', 'gender', 'age', 'majority_first', 'culture']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f'Missing required columns: {missing}')

    # Drop rows with missing key variables
    df = df.dropna(subset=['y', 'age', 'culture'])

    # Ensure types
    df['y'] = df['y'].astype(int)
    df['age'] = df['age'].astype(float)
    df['culture'] = df['culture'].astype('category')
    df['gender'] = df['gender'].astype('Int64')
    df['majority_first'] = df['majority_first'].astype('Int64')

    # Center age for interpretability (age_c)
    df['age_c'] = df['age'] - df['age'].mean()

    # Create coarse age groups for descriptive checks (not required for primary models)
    # bins chosen to reflect roughly early childhood, middle childhood, later childhood/adolescence
    df['age_group'] = pd.cut(df['age'], bins=[3, 6, 9, 12, 14], labels=['4-6', '7-9', '10-12', '13-14'], include_lowest=True)

    # Binary outcomes derived from y
    # y: 1=unchosen (undemonstrated), 2=majority, 3=minority
    df['chose_demonstrated'] = df['y'].apply(lambda v: 1 if v in [2, 3] else 0).astype(int)
    df['chose_majority'] = df['y'].apply(lambda v: 1 if v == 2 else 0).astype(int)

    # Simple binary gender indicator for models (1 = male, 0 = female). Keep original gender column as well.
    # In the data coding: 1 = girl, 2 = boy
    df['is_male'] = df['gender'].apply(lambda v: 1 if v == 2 else 0).astype(int)

    # Make a categorical view of culture suitable for modeling
    df['culture_cat'] = df['culture'].astype('category')

    # Keep and return all columns that will be used downstream
    keep_cols = list(df.columns)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Run three complementary analyses to test whether reliance on social information and preference for majority cues vary by age and culture:
      1) Multinomial logistic regression on the full multiclass outcome y (1=undemonstrated,2=majority,3=minority).
      2) Logistic regression for whether a child chose a demonstrated option (chose_demonstrated: demonstrated vs undemonstrated).
      3) Among children who chose a demonstrated option, logistic regression for preferring the majority (chose_majority).

    For (2) and (3) we fit base models and interaction models (age x culture) and perform a likelihood-ratio test to evaluate whether developmental effects differ by culture.
    Returns a dictionary with fitted model objects and LR-test results.
    """
    results = {}
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from scipy import stats

    # Prepare design matrices: create culture dummy variables (drop first to avoid multicollinearity)
    culture_dummies = pd.get_dummies(df['culture_cat'], prefix='culture', drop_first=True)

    # Base covariates for all models
    base_covs = pd.concat([df[['age_c', 'is_male', 'majority_first']].astype(float), culture_dummies], axis=1)
    base_covs = sm.add_constant(base_covs)

    # 1) Multinomial logistic regression on full outcome y
    # Use MNLogit: it models each non-reference category vs reference. We'll set reference as y==1 (undemonstrated) by recoding.
    try:
        exog = base_covs
        endog = df['y']
        # statsmodels' MNLogit expects endog coded 0..K-1; subtract 1 so reference is 0 (unchosen)
        mnlogit = sm.MNLogit(endog - 1, exog).fit(method='newton', disp=False)
        results['mnlogit'] = mnlogit
    except Exception as e:
        results['mnlogit_error'] = str(e)

    # Helper to fit logistic models and LR test comparing base vs interaction model
    def fit_and_test_logit(y_col, df_subset, test_interaction_on='age_c'):
        # Build base matrix
        culture_dm = pd.get_dummies(df_subset['culture_cat'], prefix='culture', drop_first=True)
        X_base = pd.concat([df_subset[['age_c', 'is_male', 'majority_first']].astype(float), culture_dm], axis=1)
        X_base = sm.add_constant(X_base)
        y_vec = df_subset[y_col].astype(int)

        # Fit base model
        model_base = sm.Logit(y_vec, X_base).fit(disp=False)

        # Build interaction model: age_c * each culture dummy
        # Create interaction columns between age_c and each culture dummy
        inter_cols = {}
        for c in culture_dm.columns:
            inter_cols[f'{c}_x_age'] = culture_dm[c] * df_subset['age_c']
        if len(inter_cols) > 0:
            X_int = pd.concat([X_base, pd.DataFrame(inter_cols, index=X_base.index)], axis=1)
        else:
            X_int = X_base.copy()

        model_int = sm.Logit(y_vec, X_int).fit(disp=False)

        # Likelihood ratio test
        llf_base = model_base.llf
        llf_int = model_int.llf
        lr_stat = 2 * (llf_int - llf_base)
        df_diff = model_int.df_model - model_base.df_model
        p_value = stats.chi2.sf(lr_stat, df_diff) if df_diff > 0 else np.nan

        return {
            'model_base': model_base,
            'model_interaction': model_int,
            'lr_stat': lr_stat,
            'lr_df': df_diff,
            'lr_pvalue': p_value
        }

    # 2) Logistic regression: chose_demonstrated (demonstrated vs undemonstrated)
    try:
        res_demo = fit_and_test_logit('chose_demonstrated', df)
        results['chose_demonstrated'] = res_demo
    except Exception as e:
        results['chose_demonstrated_error'] = str(e)

    # 3) Logistic regression among those who chose a demonstrated option: majority vs minority
    try:
        df_demo = df[df['chose_demonstrated'] == 1].copy()
        if df_demo.shape[0] < 30:
            # small sample warning, still attempt fit but note it
            results['chose_majority_warning'] = f'Small N in demonstrated subset: {df_demo.shape[0]} rows'
        res_major = fit_and_test_logit('chose_majority', df_demo)
        results['chose_majority'] = res_major
    except Exception as e:
        results['chose_majority_error'] = str(e)

    # Summarize key inferential outputs for easy inspection
    summary = {
        'mnlogit_params': None,
        'chose_demonstrated_lr_pvalue': None,
        'chose_majority_lr_pvalue': None
    }
    if 'mnlogit' in results and hasattr(results['mnlogit'], 'params'):
        summary['mnlogit_params'] = results['mnlogit'].params
    if 'chose_demonstrated' in results and 'lr_pvalue' in results['chose_demonstrated']:
        summary['chose_demonstrated_lr_pvalue'] = results['chose_demonstrated']['lr_pvalue']
    if 'chose_majority' in results and 'lr_pvalue' in results['chose_majority']:
        summary['chose_majority_lr_pvalue'] = results['chose_majority']['lr_pvalue']

    results['summary'] = summary

    # Interpretation note (not executed):
    # - If lr_pvalue for 'chose_demonstrated' is small (e.g., < .05), there is evidence that age effects on whether children rely on social information vary across cultures.
    # - If lr_pvalue for 'chose_majority' is small, there is evidence that age effects on preference for majority among those who copy vary across cultures.
    # - Non-significant p-values support the hypothesis that reliance and majority preference do NOT vary substantially across cultures/developmental stages (i.e., user's belief of 'No').

    return results


