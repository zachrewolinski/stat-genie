from typing import Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle

# Optional top-level read (kept from original; can be removed if not needed)
try:
    df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/positive_leading_statement_output/boxes.csv')
except Exception:
    # If file not present, leave df undefined; users should call transform() with their dataframe.
    df = None

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to produce the columns needed for modelling.

    Output columns required by the models:
      - age (numeric, years)
      - culture (kept as integer categorical)
      - is_female (1 if gender == 1 (girl), 0 if gender == 2 (boy))
      - majority_first (0/1 as in raw data)
      - is_social (1 if y == 2 or y == 3; 0 if y == 1)
      - majority_pref (1 if y == 2, 0 if y == 3; set to NaN where y == 1)

    The function drops rows with missing values in the variables required for the models.
    """
    import numpy as np
    import pandas as pd

    # Work on a copy to avoid modifying the original
    df = df.copy()

    # Ensure the key columns exist
    required_cols = ['y', 'gender', 'age', 'majority_first', 'culture']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input dataframe: {missing}")

    # Drop rows with missing values in any of the predictor/outcome variables
    df = df.dropna(subset=required_cols)

    # Ensure types
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['gender'] = pd.to_numeric(df['gender'], errors='coerce')
    df['majority_first'] = pd.to_numeric(df['majority_first'], errors='coerce')
    df['culture'] = pd.to_numeric(df['culture'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')

    # Drop any rows that became NA after coercion
    df = df.dropna(subset=['age', 'gender', 'majority_first', 'culture', 'y'])

    # Create binary gender flag: is_female = 1 if gender == 1 (girl), 0 if gender == 2 (boy)
    # If there are other codes, treat as NA and drop
    df['is_female'] = df['gender'].apply(lambda x: 1 if x == 1 else (0 if x == 2 else np.nan))
    df = df.dropna(subset=['is_female'])

    # majority_first should already be 0/1, but coerce any non-zero to 1
    df['majority_first'] = df['majority_first'].apply(lambda x: 1 if x == 1 else 0)

    # Dependent variables derived from y (1=unchosen option, 2=majority option, 3=minority option)
    # is_social: 1 if chose majority (2) or minority (3), else 0
    df['is_social'] = df['y'].apply(lambda v: 1 if v in [2, 3] else 0)

    # majority_pref: among those who used social information, 1 if majority (2), 0 if minority (3)
    # For those who did not use social information (y == 1) set to NaN; modeling code will subset appropriately.
    def maj_pref(v):
        if v == 2:
            return 1
        if v == 3:
            return 0
        return np.nan

    df['majority_pref'] = df['y'].apply(maj_pref)

    # Keep culture as-is (numeric site id). Also coerce to integer type for grouping/clustering.
    df['culture'] = df['culture'].astype(int)

    # Optional: create simple developmental stage groups for descriptive checks
    # Early: 4-6, Middle: 7-9, Late: 10-14
    def age_group(a):
        if a <= 6:
            return 'Early'
        if a <= 9:
            return 'Middle'
        return 'Late'

    df['age_group'] = df['age'].apply(age_group)

    # Final drop in case majority_pref contains all-NaN for some rows (we keep them because is_social model uses them)
    # Return the transformed dataframe containing all columns needed for modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit two logistic (binomial) models addressing the research question:
      1) Reliance on social information (is_social): logistic regression predicting whether the child used social information
         (majority or minority) vs chose the unchosen option. Predictors: age, culture, interaction age * culture, is_female, majority_first.
      2) Majority preference among social learners (majority_pref): logistic regression among rows where is_social == 1 predicting whether the
         child chose the majority option (vs minority). Same predictors as above.

    Both models return cluster-robust standard errors clustered by culture (site) to account for within-site dependence.

    Returns a dict with keys 'social_model' and 'majority_model' containing the statsmodels results objects (with cluster-robust cov).
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import numpy as np
    from statsmodels.stats.sandwich_covariance import cov_cluster, cov_hc3
    from statsmodels.tools.tools import add_constant

    results = {}

    # Ensure necessary columns are present
    needed = ['is_social', 'majority_pref', 'age', 'culture', 'is_female', 'majority_first']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Model 1: Reliance on social information (binary)
    # Use age * C(culture) interaction to test whether age effects vary across cultures.
    # Use Logit instead of GLM to ensure availability of get_robustcov_results on the results object.
    formula1 = 'is_social ~ age * C(culture) + is_female + majority_first'
    model1 = smf.logit(formula=formula1, data=df)
    try:
        res1 = model1.fit(disp=False)
    except Exception:
        # Try with a different optimizer if default fails
        res1 = model1.fit(disp=False, method='bfgs', maxiter=100)

    # Attempt to produce cluster-robust covariance (cluster by culture). Use get_robustcov_results if available.
    try:
        res1_clus = res1.get_robustcov_results(cov_type='cluster', groups=df['culture'])
    except Exception:
        # Fallback: manually compute cluster-robust covariance matrix and attach it to the results via a wrapper
        try:
            # cov_cluster expects (result, groups)
            cov = cov_cluster(res1, df['culture'])
            res1_clus = res1
            # Attach cluster covariance to the results object by setting cov_params_default and normalized_cov_params if possible
            # Many downstream methods use bse or cov_params; we'll set cov_params and bse attributes via a shallow wrapper object.
            # Create a simple namespace-like wrapper to hold the original results and override cov_params and bse.
            class ResultWrapper:
                def __init__(self, orig, cov):
                    self.orig = orig
                    self._cov = cov

                def __getattr__(self, name):
                    return getattr(self.orig, name)

                def cov_params(self):
                    return self._cov

                def bse(self):
                    # standard errors from covariance matrix
                    return np.sqrt(np.diag(self._cov))

                def summary(self, *args, **kwargs):
                    # Try to produce a summary using the original summary but replace bse if possible.
                    # Many summary routines use self.params and self.bse(); since summary is bound to orig, call it.
                    return self.orig.summary()

            res1_clus = ResultWrapper(res1, cov)
        except Exception:
            # As a last resort, use HC3 robust covariance
            try:
                res1_clus = res1.get_robustcov_results(cov_type='HC3')
            except Exception:
                res1_clus = res1  # plain results

    results['social_model'] = res1_clus

    # Model 2: Among social learners, preference for majority (binary)
    df_social = df[df['is_social'] == 1].copy()
    if df_social.shape[0] < 10:
        # Not enough data to fit model reliably
        results['majority_model'] = None
    else:
        formula2 = 'majority_pref ~ age * C(culture) + is_female + majority_first'
        model2 = smf.logit(formula=formula2, data=df_social)
        try:
            res2 = model2.fit(disp=False)
        except Exception:
            res2 = model2.fit(disp=False, method='bfgs', maxiter=100)

        try:
            res2_clus = res2.get_robustcov_results(cov_type='cluster', groups=df_social['culture'])
        except Exception:
            try:
                cov2 = cov_cluster(res2, df_social['culture'])
                class ResultWrapper2:
                    def __init__(self, orig, cov):
                        self.orig = orig
                        self._cov = cov

                    def __getattr__(self, name):
                        return getattr(self.orig, name)

                    def cov_params(self):
                        return self._cov

                    def bse(self):
                        return np.sqrt(np.diag(self._cov))

                    def summary(self, *args, **kwargs):
                        return self.orig.summary()

                res2_clus = ResultWrapper2(res2, cov2)
            except Exception:
                try:
                    res2_clus = res2.get_robustcov_results(cov_type='HC3')
                except Exception:
                    res2_clus = res2

        results['majority_model'] = res2_clus

    # Print brief summaries for convenience (users can further inspect the results objects)
    print('\n=== Reliance on social information (is_social) model summary (cluster-robust SEs) ===')
    try:
        print(results['social_model'].summary())
    except Exception:
        print('Could not print social_model summary')

    print('\n=== Majority preference among social learners (majority_pref) model summary (cluster-robust SEs) ===')
    if results['majority_model'] is None:
        print('Not enough social learners to fit majority preference model.')
    else:
        try:
            print(results['majority_model'].summary())
        except Exception:
            print('Could not print majority_model summary')

    return results