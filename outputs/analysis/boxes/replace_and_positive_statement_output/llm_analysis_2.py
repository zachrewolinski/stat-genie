from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for modeling.

    Produces the following columns required by the models:
      - SocialReliance: binary (1 if y in {2,3}, 0 if y==1)
      - MajorityChoice: binary (1 if y==2, 0 otherwise)
      - Age_z: standardized age (mean 0, sd 1)
      - Age_group: categorical developmental stage ('young','middle','older')
      - gender, majority_first, culture: carried over and validated

    Drops rows with missing values in any columns needed for the models.
    """
    df = df.copy()

    # Ensure required columns exist
    required = ['y', 'age', 'gender', 'majority_first', 'culture']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Drop rows with missing values in the required raw columns
    df = df.dropna(subset=required)

    # Derive binary outcomes
    df['SocialReliance'] = df['y'].apply(lambda v: 1 if v in [2, 3] else 0).astype(int)
    df['MajorityChoice'] = df['y'].apply(lambda v: 1 if v == 2 else 0).astype(int)

    # Standardize age (center + scale)
    # Use population std (ddof=0) for consistency with original code
    age_mean = df['age'].mean()
    age_std = df['age'].std(ddof=0)
    if age_std == 0 or np.isnan(age_std):
        # avoid division by zero; produce zeros if no variance
        df['Age_z'] = 0.0
    else:
        df['Age_z'] = (df['age'] - age_mean) / age_std

    # Create age groups for descriptive/subgroup analyses
    def age_group_label(a):
        # As specified: young (4-6), middle (7-10), older (11-14).
        # If age falls outside, still categorize sensibly.
        try:
            a_val = float(a)
        except Exception:
            return pd.NA
        if a_val <= 6:
            return 'young'
        elif a_val <= 10:
            return 'middle'
        else:
            return 'older'

    df['Age_group'] = df['age'].apply(age_group_label).astype('category')

    # Ensure culture is an integer ID. Try numeric conversion first.
    # If conversion produces NaN (non-numeric), use categorical codes.
    culture_numeric = pd.to_numeric(df['culture'], errors='coerce')
    if culture_numeric.isna().any():
        df['culture'] = pd.Categorical(df['culture']).codes.astype(int)
    else:
        # convert to integer type (safe if values are floats representing ints)
        df['culture'] = culture_numeric.astype(int)

    # Validate and standardize majority_first to 0/1
    # Accept True/False, 0/1, '0'/'1'
    def to_binary_majority_first(x):
        if pd.isna(x):
            return pd.NA
        if isinstance(x, (bool, np.bool_)):
            return int(x)
        try:
            # Try numeric conversion
            num = float(x)
            return 1 if num >= 1 else 0
        except Exception:
            # String cases
            s = str(x).strip().lower()
            if s in {'1', 'true', 't', 'yes', 'y'}:
                return 1
            else:
                return 0

    df['majority_first'] = df['majority_first'].apply(to_binary_majority_first).astype(int)

    # Gender: keep as-is (1=girl, 2=boy). Add helper is_boy for modeling convenience.
    df['is_boy'] = df['gender'].apply(lambda x: 1 if x == 2 else 0).astype(int)

    # Final subset: drop any rows that have NA in the final required columns
    final_required = ['SocialReliance', 'MajorityChoice', 'Age_z', 'Age_group', 'gender', 'majority_first', 'culture']
    df = df.dropna(subset=final_required)

    # Reset index
    df = df.reset_index(drop=True)

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Run two logistic regression models (Logit) with clustered (by culture) robust standard errors.

    Models:
      1) SocialReliance ~ Age_z * C(culture) + is_boy + majority_first
      2) MajorityChoice ~ Age_z * C(culture) + is_boy + majority_first

    Returns a dictionary with the raw fitted results objects and their clustered-robust counterparts.
    """
    results = {}

    # Ensure required columns exist in df
    for col in ['SocialReliance', 'MajorityChoice', 'Age_z', 'culture', 'is_boy', 'majority_first']:
        if col not in df.columns:
            raise ValueError(f"Required column for modeling missing: {col}")

    # Formula with interaction between Age_z and culture (culture treated as categorical)
    formula = 'OUTCOME ~ Age_z * C(culture) + is_boy + majority_first'

    def fit_clustered(outcome_col):
        f = formula.replace('OUTCOME', outcome_col)
        # Use Logit (canonical logistic regression) so we can obtain clustered robust cov easily
        model_inst = smf.logit(formula=f, data=df)
        res = model_inst.fit(disp=False)

        # Try to obtain clustered robust covariance results using the results' helper.
        try:
            clustered = res.get_robustcov_results(cov_type='cluster', groups=df['culture'])
        except Exception:
            # Fallback: compute clustered covariance matrix and attach a minimal wrapper
            cov_cluster = sm.stats.sandwich_covariance.cov_cluster(res, df['culture'])
            # Create a simple wrapper object that mimics the attributes used downstream:
            class ClusteredResults:
                def __init__(self, base_res, cov):
                    self._base = base_res
                    self.cov_cluster = cov
                    self.params = base_res.params
                    self.bse = np.sqrt(np.diag(cov))
                    # compute z and pvalues
                    with np.errstate(divide='ignore', invalid='ignore'):
                        z = self.params / self.bse
                    self.pvalues = 2 * (1 - stats.norm.cdf(np.abs(z)))
                    self._summary_text = None

                def summary(self):
                    # Produce a simple textual summary object with as_text()
                    if self._summary_text is None:
                        # construct a small table-like text
                        header = f"Clustered robust results (clusters by culture)\n"
                        rows = []
                        rows.append(f"{'param':<20}{'coef':>12}{'std err':>12}{'z':>12}{'P>|z|':>12}")
                        for name in self.params.index:
                            coef = self.params[name]
                            se = self.bse[self.params.index.get_loc(name)]
                            z = coef / se if se != 0 else np.nan
                            p = 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
                            rows.append(f"{name:<20}{coef:12.4f}{se:12.4f}{z:12.4f}{p:12.4f}")
                        self._summary_text = header + "\n".join(rows)
                    class TextHolder:
                        def __init__(self, txt):
                            self._txt = txt
                        def as_text(self):
                            return self._txt
                    return TextHolder(self._summary_text)

            clustered = ClusteredResults(res, cov_cluster)

        return {'raw': res, 'clustered': clustered}

    results['SocialReliance_model'] = fit_clustered('SocialReliance')
    results['MajorityChoice_model'] = fit_clustered('MajorityChoice')

    # Summaries: use clustered.summary().as_text() when possible
    summaries = {}
    try:
        summaries['SocialReliance_clustered_summary'] = results['SocialReliance_model']['clustered'].summary().as_text()
    except Exception:
        # Fallback to raw summary text
        summaries['SocialReliance_clustered_summary'] = results['SocialReliance_model']['clustered'].summary().as_text() if hasattr(results['SocialReliance_model']['clustered'].summary(), 'as_text') else str(results['SocialReliance_model']['clustered'])

    try:
        summaries['MajorityChoice_clustered_summary'] = results['MajorityChoice_model']['clustered'].summary().as_text()
    except Exception:
        summaries['MajorityChoice_clustered_summary'] = results['MajorityChoice_model']['clustered'].summary().as_text() if hasattr(results['MajorityChoice_model']['clustered'].summary(), 'as_text') else str(results['MajorityChoice_model']['clustered'])

    results['summaries'] = summaries

    return results