from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/positive_leading_statement_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataset for analysis. Returns a dataframe with the exact columns used in the model:
    - affairs (keeps original numeric counts)
    - children_binary (1=yes, 0=no)
    - gender_male (1=male, 0=female)
    - children_gender (interaction children_binary * gender_male)
    - age, yearsmarried, religiousness, education, occupation, rating (controls)

    Drops rows with missing values in any of these columns.
    """
    df = df.copy()

    # Ensure affairs numeric
    if 'affairs' not in df.columns:
        raise KeyError("Input dataframe must contain 'affairs' column")
    df['affairs'] = pd.to_numeric(df['affairs'], errors='coerce')

    # Map children to binary 1/0. Accept several spellings/cases.
    if 'children' not in df.columns:
        raise KeyError("Input dataframe must contain 'children' column")
    df['children_str'] = df['children'].astype(str).str.strip().str.lower()
    df['children_binary'] = df['children_str'].map({
        'yes': 1,
        'y': 1,
        '1': 1,
        'true': 1,
        'no': 0,
        'n': 0,
        '0': 0,
        'false': 0
    })

    # If mapping produced NaNs (unexpected values), try mapping categories directly
    if df['children_binary'].isna().any():
        # attempt a fallback: treat anything that contains 'y' as yes, 'n' as no
        df.loc[df['children_binary'].isna(), 'children_binary'] = (
            df.loc[df['children_binary'].isna(), 'children_str'].str.contains('y', na=False).astype(float)
        )

    # Gender -> male indicator
    if 'gender' not in df.columns:
        raise KeyError("Input dataframe must contain 'gender' column")
    df['gender_str'] = df['gender'].astype(str).str.strip().str.lower()
    df['gender_male'] = df['gender_str'].map({'male': 1, 'm': 1, 'man': 1})
    df.loc[df['gender_male'].isna(), 'gender_male'] = (~df['gender_str'].isin(['male', 'm', 'man']) & df['gender_str'].notna()).astype(int)
    # The fallback above will code non-identified values (including 'female') as 0; ensure female -> 0
    df.loc[df['gender_str'].isin(['female', 'f', 'woman']), 'gender_male'] = 0

    # Ensure numeric controls exist and coerce to numeric
    numeric_controls = ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    for col in numeric_controls:
        if col not in df.columns:
            raise KeyError(f"Input dataframe must contain '{col}' column")
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Derived moderator interaction
    df['children_gender'] = df['children_binary'] * df['gender_male']

    # Keep only rows with non-missing values in the variables we'll use
    required_cols = [
        'affairs', 'children_binary', 'gender_male', 'children_gender'
    ] + numeric_controls
    df = df.dropna(subset=required_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    # Final dataframe includes only the columns used in modeling (keeps original affairs)
    final_cols = ['affairs', 'children_binary', 'gender_male', 'children_gender'] + numeric_controls
    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Run a set of models to estimate the association between having children and extramarital affairs,
    controlling for demographic and marriage-related covariates. Returns a dictionary of fitted model objects:
      - 'negative_binomial': Negative binomial regression (appropriate for over-dispersed counts)
      - 'ols_robust': OLS regression with robust (HC3) standard errors as a baseline
      - 'zero_inflated_poisson': Zero-inflated Poisson (attempted; may be None if estimation fails)

    Model specification (primary):
      affairs ~ children_binary + gender_male + children_gender + age + yearsmarried + religiousness + education + occupation + rating

    Interpret the coefficient on children_binary as the association of having children with the expected count of affairs
    (NB: for NB/Poisson coefficients are in log-count space; exponentiate to get multiplicative effect).
    """
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    import warnings

    # Import count models; these imports may fail on very old statsmodels versions, handle gracefully
    try:
        from statsmodels.discrete.discrete_model import NegativeBinomial
    except Exception:
        NegativeBinomial = None
    try:
        from statsmodels.discrete.count_model import ZeroInflatedPoisson
    except Exception:
        ZeroInflatedPoisson = None

    # Prepare X and y
    X_cols = ['children_binary', 'gender_male', 'children_gender', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    X = df[X_cols]
    X = sm.add_constant(X)
    y = df['affairs']

    results = {}

    # 1) Negative Binomial (preferred for over-dispersed counts)
    if NegativeBinomial is not None:
        try:
            nb_mod = NegativeBinomial(y, X)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ConvergenceWarning)
                nb_res = nb_mod.fit(disp=False)
            results['negative_binomial'] = nb_res
        except Exception as e:
            results['negative_binomial'] = f'failed: {e}'
    else:
        results['negative_binomial'] = 'NegativeBinomial not available in this statsmodels installation'

    # 2) OLS with robust standard errors (baseline linear approximation)
    try:
        ols_mod = sm.OLS(y, X)
        ols_res = ols_mod.fit(cov_type='HC3')
        results['ols_robust'] = ols_res
    except Exception as e:
        results['ols_robust'] = f'failed: {e}'

    # 3) Zero-inflated Poisson as an alternative (some respondents have zero affairs -> extra zeros)
    if ZeroInflatedPoisson is not None:
        try:
            # Use same exog for count and inflation parts, but inflation could be more parsimonious
            zip_mod = ZeroInflatedPoisson(endog=y, exog=X, exog_infl=X[['children_binary', 'gender_male', 'age']])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ConvergenceWarning)
                zip_res = zip_mod.fit(disp=False)
            results['zero_inflated_poisson'] = zip_res
        except Exception as e:
            results['zero_inflated_poisson'] = f'failed: {e}'
    else:
        results['zero_inflated_poisson'] = 'ZeroInflatedPoisson not available in this statsmodels installation'

    # Provide a small helper summary (text) for quick inspection: coefficient on children_binary
    try:
        def coef_summary(res, name):
            if isinstance(res, str):
                return res
            try:
                coef = res.params.get('children_binary')
                se = res.bse.get('children_binary')
                pval = res.pvalues.get('children_binary')
                return {'coef': float(coef), 'se': float(se), 'pval': float(pval)}
            except Exception:
                return 'could not extract children_binary coeff'

        results['summary_children_binary'] = {
            'negative_binomial': coef_summary(results.get('negative_binomial'), 'negative_binomial'),
            'ols_robust': coef_summary(results.get('ols_robust'), 'ols_robust'),
            'zero_inflated_poisson': coef_summary(results.get('zero_inflated_poisson'), 'zero_inflated_poisson')
        }
    except Exception:
        results['summary_children_binary'] = 'failed to build summary'

    return results


