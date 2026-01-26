from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/shuffle_names_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Crofoot intergroup contest data into an analysis-ready dataframe.

    Produces the following new/renamed columns used in the model:
      - focal_size: number of individuals in the focal group (from 'f_focal')
      - other_size: number of individuals in the other group (from 'f_other')
      - group_size_diff: focal_size - other_size
      - size_ratio: focal_size / other_size
      - focal_dist: distance (m) of focal group from its home-range center (from 'win')
      - other_dist: distance (m) of other group from its home-range center (from 'm_focal')
      - contest_location: categorical ('FocalHome', 'OtherHome', 'Neutral')
      - contest_location_*: dummy columns for contest_location (drop_first=True)
      - n_males_focal: number of males in focal group (from 'n_focal')
      - n_males_other: number of males in other group (from 'other')

    Notes/assumptions: variable names in the source file have inconsistent textual descriptions; this transform uses the numeric columns by name as provided.
    """
    df = df.copy()

    # Keep rows with the key variables present
    req_cols = ['dyad', 'f_focal', 'f_other', 'win', 'm_focal', 'n_focal', 'other', 'm_other']
    df = df.dropna(subset=req_cols)

    # Create numeric group size variables. We interpret 'f_focal' as focal group size and 'f_other' as other group size.
    df['focal_size'] = pd.to_numeric(df['f_focal'], errors='coerce')
    df['other_size'] = pd.to_numeric(df['f_other'], errors='coerce')

    # Absolute and relative size metrics
    df['group_size_diff'] = df['focal_size'] - df['other_size']
    # avoid division by zero
    df['size_ratio'] = df['focal_size'] / df['other_size'].replace({0: np.nan})

    # Distances from home-range centers: use 'win' as focal distance and 'm_focal' as other distance (as per dataset numeric ranges)
    df['focal_dist'] = pd.to_numeric(df['win'], errors='coerce')
    df['other_dist'] = pd.to_numeric(df['m_focal'], errors='coerce')

    # Contest location: closer to focal group's center => FocalHome; closer to other group's center => OtherHome; equal (or near equal) => Neutral
    # Use a small threshold (1 meter) to treat ties as Neutral.
    df['contest_location'] = np.where(df['focal_dist'] + 1 < df['other_dist'], 'FocalHome',
                                      np.where(df['other_dist'] + 1 < df['focal_dist'], 'OtherHome', 'Neutral'))

    # Create dummies for contest_location and drop first to avoid multicollinearity (reference = first alphabetical or remaining category depending on presence).
    dummies = pd.get_dummies(df['contest_location'], prefix='contest_location', drop_first=True)
    df = pd.concat([df, dummies], axis=1)

    # Ensure outcome is binary integer
    df['dyad'] = pd.to_numeric(df['dyad'], errors='coerce').astype('Int64')

    # Male counts controls
    df['n_males_focal'] = pd.to_numeric(df['n_focal'], errors='coerce')
    df['n_males_other'] = pd.to_numeric(df['other'], errors='coerce')

    # Final drop of any rows that still have NA in required model columns
    model_cols = ['dyad', 'size_ratio', 'group_size_diff', 'focal_dist', 'other_dist', 'n_males_focal', 'n_males_other', 'm_other']
    # also include any contest dummy columns that were created
    contest_dummy_cols = [c for c in df.columns if c.startswith('contest_location_')]
    model_cols += contest_dummy_cols

    df = df.dropna(subset=model_cols)

    # Convert m_other to integer ID for clustering purposes
    df['m_other'] = df['m_other'].astype(int)

    # Return the transformed dataframe (keeps original columns plus the engineered variables)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a binomial (logistic) regression predicting the probability the focal group wins (dyad == 1).

    Model specification (primary):
      dyad ~ size_ratio * contest_location + group_size_diff + focal_dist + other_dist + n_males_focal + n_males_other

    contest_location enters via its dummy columns (prefix 'contest_location_'); interaction with size_ratio is included.
    Standard errors are clustered by dyad-pair identifier 'm_other' when possible.

    Returns the fitted statsmodels results object.
    """
    import statsmodels.api as sm

    df = df.copy()

    # Response
    y = df['dyad'].astype(int)

    # Base predictors
    predictors = ['size_ratio', 'group_size_diff', 'focal_dist', 'other_dist', 'n_males_focal', 'n_males_other']

    # Add contest-location dummies (if any): e.g., contest_location_OtherHome, contest_location_Neutral
    contest_cols = [c for c in df.columns if c.startswith('contest_location_')]
    predictors += contest_cols

    # Add interaction terms between size_ratio and each contest-location dummy
    # Create interaction columns in the dataframe (explicitly) so they appear in the model results with clear names
    for c in contest_cols:
        inter_name = f'size_ratio_x_{c}'
        df[inter_name] = df['size_ratio'] * df[c]
        predictors.append(inter_name)

    X = df[predictors]
    X = sm.add_constant(X)

    # Fit GLM with binomial family (logit link) and attempt clustered SE by 'm_other'
    family = sm.families.Binomial()
    glm_model = sm.GLM(y, X, family=family)

    try:
        # Clustered standard errors by dyad pair (m_other)
        results = glm_model.fit(cov_type='cluster', cov_kwds={'groups': df['m_other']})
    except Exception:
        # Fallback to default fitting if clustering fails
        results = glm_model.fit()

    # Print summary for convenience and return the results object
    print(results.summary())
    return results


