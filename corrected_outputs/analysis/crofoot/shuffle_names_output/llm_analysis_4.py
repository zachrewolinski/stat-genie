from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/crofoot/shuffle_names_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Drop rows missing key columns required for predictors/outcome
    df = df.dropna(subset=['dyad', 'f_other', 'f_focal', 'win', 'm_focal'])

    # Interpret columns (use columns that are described as total group sizes):
    # According to the dataset descriptions, 'f_other' was given as "Number of individuals in focal group"
    # and 'f_focal' as "Number of individuals in other group". We create clear final columns focal_total/other_total
    df['focal_total'] = pd.to_numeric(df['f_other'], errors='coerce')
    df['other_total'] = pd.to_numeric(df['f_focal'], errors='coerce')

    # Relative size metrics
    df['rel_size'] = df['focal_total'] - df['other_total']
    # ratio (guard against division by zero)
    df['rel_size_ratio'] = df['focal_total'] / (df['other_total'].replace({0: np.nan}))
    # log ratio (stabilizes skew, add small constant to avoid log(0))
    df['log_rel_size'] = np.log(df['focal_total'] + 0.1) - np.log(df['other_total'] + 0.1)

    # Distances to home-range centers: use 'win' as distance from focal group's center and 'm_focal' as distance from other group's center
    df['dist_from_focal_home'] = pd.to_numeric(df['win'], errors='coerce')
    df['dist_from_other_home'] = pd.to_numeric(df['m_focal'], errors='coerce')

    # Define Location categorical variable
    # If contest location is appreciably closer to focal group's home center -> 'FocalHome'
    # If appreciably closer to other group's home center -> 'OtherHome'
    # If similar (within threshold) -> 'Boundary'
    threshold_m = 50.0
    cond_focal = df['dist_from_focal_home'] + threshold_m < df['dist_from_other_home']
    cond_other = df['dist_from_other_home'] + threshold_m < df['dist_from_focal_home']
    df.loc[cond_focal, 'Location'] = 'FocalHome'
    df.loc[cond_other, 'Location'] = 'OtherHome'
    df.loc[~(cond_focal | cond_other), 'Location'] = 'Boundary'
    df['Location'] = pd.Categorical(df['Location'], categories=['FocalHome', 'Boundary', 'OtherHome'])

    # Controls: male counts
    # 'n_focal' already provided (number of males in focal group) - coerce to numeric
    df['n_focal'] = pd.to_numeric(df['n_focal'], errors='coerce')
    # column 'other' in original dataset described as number of males in other group -> rename for clarity
    df['other_males'] = pd.to_numeric(df['other'], errors='coerce')

    # Sum of sizes as a control for contest scale
    df['total_size'] = df['focal_total'] + df['other_total']

    # Outcome: ensure binary int 0/1
    df['dyad'] = pd.to_numeric(df['dyad'], errors='coerce').astype('Int64')

    # Final drop of any rows with NA in variables that will be used in the model
    required_for_model = ['dyad', 'log_rel_size', 'Location', 'n_focal', 'other_males', 'total_size', 'm_other']
    df = df.dropna(subset=required_for_model)

    # Convert dyad to integer 0/1
    df['dyad'] = df['dyad'].astype(int)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    import statsmodels.formula.api as smf

    # Formula: test main effects of relative group size (log ratio) and contest location,
    # plus their interaction. Control for male composition and total size.
    # dyad is binary -> logistic regression (logit)
    formula = 'dyad ~ log_rel_size * C(Location) + n_focal + other_males + total_size'

    # Fit logistic regression using statsmodels formula API
    logit_model = smf.logit(formula, data=df).fit(disp=False)

    # Use clustered robust standard errors clustered by dyad-pair identifier 'm_other'
    # If clustering fails for any reason, return the plain fitted model
    try:
        result = logit_model.get_robustcov_results(cov_type='cluster', groups=df['m_other'])
    except Exception:
        result = logit_model

    return result


