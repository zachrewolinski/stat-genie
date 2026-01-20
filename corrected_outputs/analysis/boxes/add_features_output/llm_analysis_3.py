from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from types import SimpleNamespace
from statsmodels.stats.sandwich_covariance import cov_cluster, cov_hc3

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/boxes/add_features_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy to avoid modifying the original
    df = df.copy()

    # Keep rows with essential variables present
    df = df.dropna(subset=['y', 'age', 'culture'])

    # Create binary dependent variable: 1 if child chose the majority option (y == 2), otherwise 0
    df['ChoseMajority'] = (df['y'] == 2).astype(int)

    # Center age for interpretability (mean-centering)
    df['Age_c'] = df['age'] - df['age'].mean()

    # Ensure culture is categorical (will be used as a factor / moderator in the model)
    df['culture'] = df['culture'].astype('category')

    # Ensure gender is present and has consistent dtype
    # If gender is missing, drop those rows (control variable required)
    if 'gender' in df.columns:
        df = df.dropna(subset=['gender'])
        # Keep the original coding but ensure a stable dtype
        df['gender'] = df['gender'].astype('category')
    else:
        # If gender absent, create a missing category (but keep column name)
        df['gender'] = 'missing'
        df['gender'] = df['gender'].astype('category')

    # majority_first should be binary (0/1). If present, coerce and drop NA.
    if 'majority_first' in df.columns:
        df = df.dropna(subset=['majority_first'])
        # coerce to integer (if values are boolean or floats)
        df['majority_first'] = df['majority_first'].astype(int)
    else:
        # If the column does not exist, create it as 0s
        df['majority_first'] = 0

    # Ensure school is present for clustering. If missing, create a placeholder identifier.
    if 'school' in df.columns:
        df = df.dropna(subset=['school'])
        df['school'] = df['school'].astype(str)
    else:
        df['school'] = 'unknown_school'

    # Return dataframe with all columns needed for modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Model the probability of choosing the majority option as a function of age, culture (as moderator), and controls.
    # Uses logistic regression with an Age x Culture interaction and clustered SEs by school.

    # Formula: main effect of age (centered), culture as categorical moderator, their interaction, and controls
    formula = 'ChoseMajority ~ Age_c * C(culture) + C(gender) + majority_first'

    # Fit logistic regression (maximum likelihood)
    logit_mod = smf.logit(formula=formula, data=df)
    logit_res = logit_mod.fit(disp=False)

    # Attempt to compute cluster-robust covariance (cluster on school).
    # If clustering fails, fallback to HC3 robust covariance.
    try:
        groups = df['school'].values
        cov = cov_cluster(logit_res, groups)
    except Exception:
        cov = cov_hc3(logit_res)

    # Compute robust standard errors from the covariance matrix
    bse = np.sqrt(np.diag(cov))

    # Build a lightweight results-like object exposing common attributes
    clustered_res = SimpleNamespace(
        params=logit_res.params,
        bse=pd.Series(bse, index=logit_res.params.index),
        cov_params=pd.DataFrame(cov, index=logit_res.params.index, columns=logit_res.params.index),
        model=logit_res.model,
        result=logit_res,
        summary=lambda: logit_res.summary()
    )

    return clustered_res