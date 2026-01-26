from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster
from scipy import stats


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the analysis dataframe. Input columns expected:
      - feature1: outcome (1=unchosen option, 2=majority option, 3=minority option)
      - feature2: gender (1=girl, 2=boy)
      - feature3: age in years (4-14)
      - feature4: majority demonstrated first (0/1)
      - feature5: site id (1..8)

    Adds the following columns used in modeling:
      - Age, Age_c, Age_c2, AgeGroup, Site,
      - GenderFemale, MajorityFirst,
      - DemonstratedChosen (1 if chose majority or minority),
      - MajorityChosen (1 if chose majority; defined for all rows, used conditional on DemonstratedChosen==1)
    """
    # Work on a copy
    df = df.copy()

    # Ensure required columns exist
    required = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataframe: {missing}")

    # Drop rows with missing key variables (outcome, age, gender, order, site)
    df = df.dropna(subset=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])

    # Outcome mapping as labels (not strictly necessary but helpful)
    outcome_map = {1: 'unchosen', 2: 'majority', 3: 'minority'}
    df['Outcome'] = df['feature1'].map(outcome_map)

    # Binary: did child choose a demonstrated option (majority or minority) vs unchosen option
    df['DemonstratedChosen'] = df['feature1'].apply(lambda x: 1 if x in [2, 3] else 0).astype(int)

    # Among all cases, indicator for majority choice (1 if majority, 0 otherwise)
    df['MajorityChosen'] = df['feature1'].apply(lambda x: 1 if x == 2 else 0).astype(int)

    # Age (raw), centered and quadratic
    df['Age'] = pd.to_numeric(df['feature3'], errors='coerce')
    # Drop rows where age parsing failed
    df = df.dropna(subset=['Age'])
    df['Age_c'] = df['Age'] - df['Age'].mean()
    df['Age_c2'] = df['Age_c'] ** 2

    # Coarse age groups for descriptive checks / potential moderator
    bins = [3, 6, 9, 12, 15]  # edges: (3,6], (6,9], (9,12], (12,15]
    labels = ['4-6', '7-9', '10-12', '13-14']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)

    # Gender: create Female indicator (1=girl, 0=boy)
    # Assumes feature2 coding 1=girl, 2=boy
    df['GenderFemale'] = df['feature2'].apply(lambda x: 1 if x == 1 else 0).astype(int)

    # Majority first (procedure) ensure 0/1 int
    df['MajorityFirst'] = df['feature4'].astype(int)

    # Site as categorical (string) to be used for dummies in modeling
    df['Site'] = df['feature5'].astype(str)

    # Final minimal drop: ensure no remaining NA in model columns
    model_cols = ['DemonstratedChosen', 'MajorityChosen', 'Age', 'Age_c', 'Age_c2', 'AgeGroup', 'GenderFemale', 'MajorityFirst', 'Site']
    df = df.dropna(subset=model_cols)

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Run two logistic regression models to answer the research question:
      1) Reliance on social information: logistic regression predicting DemonstratedChosen (1 = chose a demonstrated option) using Age (linear + quadratic), Gender, MajorityFirst, and site fixed effects.
      2) Preference for majority cues: among cases where a demonstrated option was chosen, logistic regression predicting MajorityChosen (1 = majority) with the same predictor set.

    Returns a dictionary with fitted model result objects and text summaries (cluster-robust SE clustered by Site).
    """
    results = {}

    # Helper to construct a lightweight cluster-robust "result" object with a .summary().as_text() method
    class ClusterRobustResult:
        def __init__(self, params, cov, nobs, model_name="Model"):
            self.params = np.asarray(params)
            self.cov = np.asarray(cov)
            self.nobs = nobs
            self.bse = np.sqrt(np.diag(self.cov))
            # compute z and p-values (two-sided)
            # handle zeros in bse to avoid div by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                self.zvalues = np.where(self.bse > 0, self.params / self.bse, np.nan)
            self.pvalues = 2 * stats.norm.sf(np.abs(self.zvalues))
            self.model_name = model_name

        def summary(self):
            # Create a simple text summary with coef table
            header = f"Cluster-robust results ({self.model_name})\n"
            rows = []
            names = [f"b{i}" for i in range(len(self.params))]
            # If params have index-like names, try to preserve them if attached as type with names
            try:
                if hasattr(self.params, 'index'):
                    names = list(self.params.index)
            except Exception:
                pass
            # Build DataFrame for neat formatting
            df_table = pd.DataFrame({
                'coef': self.params,
                'std err': self.bse,
                'z': self.zvalues,
                'P>|z|': self.pvalues
            }, index=names)
            txt = header + df_table.to_string(float_format=lambda x: f"{x:0.4f}")
            class TextWrap:
                def __init__(self, s):
                    self._s = s
                def as_text(self):
                    return self._s
            return TextWrap(txt)

    # Ensure Site is string categorical
    df = df.copy()
    df['Site'] = df['Site'].astype(str)

    # Build site dummies (fixed effects), drop first to avoid collinearity
    site_dummies = pd.get_dummies(df['Site'], prefix='Site', drop_first=True)

    # Predictor matrix common columns
    predictors = pd.concat([
        df[['Age_c', 'Age_c2', 'GenderFemale', 'MajorityFirst']].astype(float),
        site_dummies
    ], axis=1)

    # Add intercept
    predictors = sm.add_constant(predictors, has_constant='add')

    # 1) Model: DemonstratedChosen ~ predictors
    y1 = df['DemonstratedChosen'].astype(int)
    model1 = sm.Logit(y1, predictors)
    try:
        res1 = model1.fit(disp=False)
        # Compute cluster-robust covariance by Site
        cov1 = cov_cluster(res1, df['Site'].values)
        res1_clust = ClusterRobustResult(res1.params, cov1, res1.nobs, model_name="DemonstratedChosen")
    except Exception:
        # fallback: use GLM with binomial family (more numerically stable)
        glm1 = sm.GLM(y1, predictors, family=sm.families.Binomial())
        res1 = glm1.fit()
        # Compute cluster-robust covariance by Site
        cov1 = cov_cluster(res1, df['Site'].values)
        res1_clust = ClusterRobustResult(res1.params, cov1, res1.nobs, model_name="DemonstratedChosen")

    results['demonstrated_model'] = {
        'fitted_model': res1,
        'cluster_robust_model': res1_clust,
        'summary_text': res1_clust.summary().as_text()
    }

    # 2) Model among those who chose a demonstrated option: Majority vs Minority
    df_demo = df[df['DemonstratedChosen'] == 1].copy()
    if df_demo.shape[0] < 10:
        # Not enough cases to fit a model
        results['majoritypref_model'] = {'error': 'Too few demonstrated-choice cases to fit model', 'n': df_demo.shape[0]}
        # also return predictors columns used in the first model
        results['predictors_columns'] = list(predictors.columns)
        return results

    site_dummies_demo = pd.get_dummies(df_demo['Site'], prefix='Site', drop_first=True)
    predictors_demo = pd.concat([
        df_demo[['Age_c', 'Age_c2', 'GenderFemale', 'MajorityFirst']].astype(float),
        site_dummies_demo
    ], axis=1)
    predictors_demo = sm.add_constant(predictors_demo, has_constant='add')
    y2 = df_demo['MajorityChosen'].astype(int)

    model2 = sm.Logit(y2, predictors_demo)
    try:
        res2 = model2.fit(disp=False)
        cov2 = cov_cluster(res2, df_demo['Site'].values)
        res2_clust = ClusterRobustResult(res2.params, cov2, res2.nobs, model_name="MajorityChosen")
    except Exception:
        glm2 = sm.GLM(y2, predictors_demo, family=sm.families.Binomial())
        res2 = glm2.fit()
        cov2 = cov_cluster(res2, df_demo['Site'].values)
        res2_clust = ClusterRobustResult(res2.params, cov2, res2.nobs, model_name="MajorityChosen")

    results['majoritypref_model'] = {
        'fitted_model': res2,
        'cluster_robust_model': res2_clust,
        'summary_text': res2_clust.summary().as_text(),
        'n_demonstrated': df_demo.shape[0]
    }

    # Optional: return the predictors used for quick inspection
    results['predictors_columns'] = list(predictors.columns)

    return results