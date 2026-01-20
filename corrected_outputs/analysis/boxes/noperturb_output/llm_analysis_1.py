from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# If you want to run transform on a file, set the path here (not required for import)
# df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/boxes/noperturb_output/boxes.csv')


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw cross-cultural children's dataset for modeling.

    Produces the following required columns used in the model:
      - MajorityChoice : binary (1 if y == 2 (majority), else 0)
      - age_c          : age centered around the sample mean
      - age2           : squared centered age (nonlinear term)
      - culture        : site ID preserved as categorical (used with C(culture) in formula)
      - gender_male    : 1 if gender == 2 (boy), 0 if gender == 1 (girl)
      - majority_first : numeric 0/1 indicator preserved from the original data

    The function also drops rows missing any of the variables necessary for modeling.
    """
    # make a copy to avoid modifying original
    df = df.copy()

    # Coerce key columns to numeric where applicable to ensure proper NA handling
    # and consistent comparisons. Do this before dropping missing values.
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
    if 'y' in df.columns:
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
    if 'majority_first' in df.columns:
        df['majority_first'] = pd.to_numeric(df['majority_first'], errors='coerce')

    # Keep only rows with required data
    required_cols = ['y', 'age', 'culture', 'gender', 'majority_first']
    df = df.dropna(subset=required_cols)

    # Dependent variable: majority choice (1 if y == 2, else 0)
    df['MajorityChoice'] = (df['y'] == 2).astype(int)

    # Center age and add quadratic term to capture non-linear development
    age_mean = df['age'].mean()
    df['age_c'] = df['age'] - age_mean
    df['age2'] = df['age_c'] ** 2

    # Ensure culture is preserved (as category) for use in formula and clustering
    # keep original IDs but cast to category for clarity
    df['culture'] = df['culture'].astype('category')

    # Gender: original coding 1=girl, 2=boy -> create male indicator
    df['gender_male'] = (df['gender'] == 2).astype(int)

    # majority_first should be binary; coerce to integer 0/1 (assumes original already 0/1 or similar)
    # Any non-binary numeric values will be kept as-is; rows with NA already removed.
    df['majority_first'] = df['majority_first'].astype(int)

    # Drop any rows that became NA during coercion or transformations for model columns
    final_required = ['MajorityChoice', 'age_c', 'age2', 'culture', 'gender_male', 'majority_first']
    df = df.dropna(subset=final_required)

    # Reset index for a clean dataframe
    df = df.reset_index(drop=True)

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression predicting probability of choosing the majority option.

    Model specification (fixed-effects with clustered SEs by culture):
      MajorityChoice ~ age_c + age2 + C(culture) + age_c:C(culture) + gender_male + majority_first

    The interaction age_c:C(culture) allows the age slope to differ across cultures.
    Clustered standard errors are computed clustering on culture.

    Returns a statsmodels results object with clustered robust covariance if supported by fit.
    """
    # Formula: main effects of (centered) age and age^2, culture as factor, interaction of age with culture,
    # and controls gender_male and majority_first.
    formula = (
        'MajorityChoice ~ age_c + age2 + C(culture) + age_c:C(culture) '
        '+ gender_male + majority_first'
    )

    # Prepare cluster groups as an array of integer codes for reliability
    # Ensure culture is categorical; if not, convert temporarily
    if not pd.api.types.is_categorical_dtype(df['culture']):
        cluster_groups = pd.Categorical(df['culture']).codes
    else:
        cluster_groups = df['culture'].cat.codes

    # Fit the binomial logistic regression with clustered covariance requested at fit time.
    # Many statsmodels models accept cov_type and cov_kwds in fit(); when supported this
    # will compute and attach clustered robust standard errors to the results object.
    try:
        model_inst = smf.logit(formula=formula, data=df)
        model_fit = model_inst.fit(disp=False, cov_type='cluster', cov_kwds={'groups': cluster_groups})
    except TypeError:
        # Fallback: if the fit method does not accept cov_type (older versions),
        # fit normally and return the fitted results object. Users can compute clustered
        # covariances externally if needed.
        model_fit = smf.logit(formula=formula, data=df).fit(disp=False)

    return model_fit