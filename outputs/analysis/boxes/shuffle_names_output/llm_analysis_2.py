from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/shuffle_names_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe with the exact columns used in the statistical models.

    Input schema (expected columns):
      - majority_first: outcome code with values {1=unchosen option, 2=majority option, 3=minority option}
      - culture: (in this dataset) child's age in years (numeric)
      - age: (in this dataset) indicator whether the majority option was demonstrated first (0/1)
      - gender: 1=girl, 2=boy
      - y: site ID (1..8)

    Output columns (kept/created):
      - MajorityChoice: binary (1 if majority chosen, else 0)
      - age_years: child's age in years (float)
      - age_c: centered age (age_years - mean(age_years)) [helper column]
      - age_sq: squared centered age (for nonlinearity if desired) [helper column]
      - is_majority_first: 0/1 indicator whether majority was demonstrated first
      - gender_female: 1 if girl, 0 if boy
      - site_id: string-coded site identifier
    """
    df = df.copy()

    # Drop rows with missing essential variables
    df = df.dropna(subset=['majority_first', 'culture', 'age', 'gender', 'y'])

    # Dependent variable: did the child choose the majority-demonstrated option?
    df['MajorityChoice'] = (df['majority_first'] == 2).astype(int)

    # Age: the column 'culture' contains the child's age in years (per provided schema)
    df['age_years'] = df['culture'].astype(float)
    # Center age for interpretability and add quadratic term to allow nonlinearity (helper columns)
    df['age_c'] = df['age_years'] - df['age_years'].mean()
    df['age_sq'] = df['age_c'] ** 2

    # Experimental control: whether majority was demonstrated first.
    df['is_majority_first'] = df['age'].astype(int)

    # Gender: encode as female indicator (1 = girl, 0 = boy)
    df['gender_female'] = (df['gender'] == 1).astype(int)

    # Site / cultural context
    df['site_id'] = df['y'].astype(str)

    # Keep only the columns needed for modeling (and allowed helper columns)
    out_cols = ['MajorityChoice', 'age_years', 'age_c', 'age_sq', 'is_majority_first', 'gender_female', 'site_id']
    return df[out_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit population-averaged logistic regression models for the binary outcome 'MajorityChoice'.

    Two models are estimated:
      1) Main effects model: MajorityChoice ~ age_years + gender_female + is_majority_first + C(site_id)
         - site_id entered as categorical fixed effects to control for baseline cross-cultural differences.
         - Cluster-robust standard errors are computed by site (to account for within-site dependence).
           Implemented here using GEE with an exchangeable working correlation (population-averaged model).

      2) Interaction model: MajorityChoice ~ age_years * C(site_id) + gender_female + is_majority_first
         - Tests whether the age slope differs across sites (i.e., whether developmental trajectories differ by culture).

    Returns a dict with keys 'main' and 'interaction' containing fitted results objects.
    """
    # Ensure no missing values in model columns
    df = df.dropna(subset=['MajorityChoice', 'age_years', 'gender_female', 'is_majority_first', 'site_id'])

    # Use GEE (population-averaged model) with exchangeable covariance to obtain cluster-robust SEs by site
    cov_struct = sm.cov_struct.Exchangeable()

    formula_main = 'MajorityChoice ~ age_years + gender_female + is_majority_first + C(site_id)'
    gee_main = smf.gee(formula_main, groups="site_id", data=df, family=sm.families.Binomial(), cov_struct=cov_struct)
    res_main = gee_main.fit()

    formula_int = 'MajorityChoice ~ age_years * C(site_id) + gender_female + is_majority_first'
    gee_int = smf.gee(formula_int, groups="site_id", data=df, family=sm.families.Binomial(), cov_struct=cov_struct)
    res_int = gee_int.fit()

    return {
        'main': res_main,
        'interaction': res_int
    }