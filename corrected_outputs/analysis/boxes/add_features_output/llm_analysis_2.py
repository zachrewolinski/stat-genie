from typing import Any
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Keep only rows with non-missing outcome and key predictors
    df = df.dropna(subset=['y', 'age', 'culture'])

    # Construct the binary outcome: 1 if child chose the majority option (y==2), else 0
    df['MajorityChoice'] = (df['y'] == 2).astype(int)

    # Create a mean-centered age variable for interpretability
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    age_mean = df['age'].mean()
    df['age_c'] = df['age'] - age_mean

    # Create female indicator: original coding 1 = girl, 2 = boy
    if 'gender' in df.columns:
        df['female'] = (df['gender'] == 1).astype(int)
    else:
        # If gender not available, create NA female column so downstream checks can catch it
        df['female'] = pd.NA

    # Ensure majority_first is numeric binary (0/1)
    if 'majority_first' in df.columns:
        df['majority_first'] = pd.to_numeric(df['majority_first'], errors='coerce').fillna(0).astype(int)
    else:
        df['majority_first'] = 0

    # Ensure culture is categorical (keep original numeric IDs but cast to category)
    df['culture'] = df['culture'].astype('category')

    # Ensure school exists as a categorical variable for clustering (fill missing with 'UNK')
    if 'school' in df.columns:
        df['school'] = df['school'].fillna('UNK').astype(str)
    else:
        # If school not available, create a dummy clustering variable using culture (less ideal)
        df['school'] = df['culture'].astype(str)

    # Optionally drop rows with missing constructed columns
    df = df.dropna(subset=['MajorityChoice', 'age_c', 'female', 'majority_first', 'culture', 'school'])

    return df


def model(df: pd.DataFrame) -> Any:
    # Formula: test main effect of age and whether the age effect differs across cultures
    # Controls: gender (female) and majority_first. Culture enters as a factor and interacts with age.
    formula = 'MajorityChoice ~ age_c + C(culture) + age_c:C(culture) + female + majority_first'

    # Fit a logistic regression (GLM with binomial family)
    glm_mod = smf.glm(formula=formula, data=df, family=sm.families.Binomial())

    # Try to fit with clustered covariances by re-fitting with cov_type if supported.
    # Some versions of statsmodels support passing cov_type to fit; this will produce results
    # whose covariance reflects the requested cov_type. If that fails, fall back to HC1,
    # then to the default fit.
    try:
        glm_res = glm_mod.fit(cov_type='cluster', cov_kwds={'groups': df['school']})
    except Exception:
        try:
            glm_res = glm_mod.fit(cov_type='HC1')
        except Exception:
            glm_res = glm_mod.fit()

    return glm_res