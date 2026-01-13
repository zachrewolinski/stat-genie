from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/add_features_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataset for modeling the developmental trajectory of majority reliance across cultures.
    Produces the following new columns used in modeling:
      - MajorityChoice: binary outcome (1 if y==2 (majority), else 0)
      - age_c: age centered around the sample mean
      - age_c2: squared term for age_c to capture nonlinearity
      - IsBoy: 1 if gender==2, else 0
      - culture: converted to categorical (keeps original codes but as category dtype)
      - school: school identifier as string (used for clustering)
    Also drops rows with missing values in variables required for the model.
    """
    # Work on a copy
    df = df.copy()

    # Required columns for analysis
    required = ['y', 'age', 'culture', 'gender', 'majority_first', 'religiousness', 'calworks', 'school']
    # Drop rows missing any required column
    df = df.dropna(subset=required)

    # Create binary dependent variable: 1 if chose majority (y == 2), else 0
    df['MajorityChoice'] = (df['y'] == 2).astype(int)

    # Center age for interpretability and create quadratic term to allow non-linear growth
    df['age_c'] = df['age'] - df['age'].mean()
    df['age_c2'] = df['age_c'] ** 2

    # Gender: create male indicator (IsBoy = 1 for boys (gender==2), 0 for girls)
    df['IsBoy'] = (df['gender'] == 2).astype(int)

    # Ensure majority_first is numeric 0/1
    df['majority_first'] = pd.to_numeric(df['majority_first'], errors='coerce').astype(int)

    # Keep culture as categorical for modeling with factor notation
    df['culture'] = df['culture'].astype('category')

    # Ensure school is treated as an identifier (string) for clustering
    df['school'] = df['school'].astype(str)

    # Final safety drop in case any new columns have NAs
    model_cols = ['MajorityChoice', 'age_c', 'age_c2', 'culture', 'IsBoy', 'majority_first', 'religiousness', 'calworks', 'school']
    df = df.dropna(subset=model_cols)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression predicting the probability of choosing the majority option.
    The model includes an interaction between centered age and culture to test whether
    developmental trajectories differ across cultural contexts. Controls include gender,
    demonstration order, religiousness, and calworks (SES proxy). Standard errors are
    clustered by school to account for within-school dependence.

    Returns a dictionary with:
      - 'fit': the fitted (cluster-robust) results object if available (or the regular results object)
      - 'model_raw': the raw fitted LogitResults (before clustering if clustering was applied at fit)
      - 'predicted_prob_grid': a DataFrame with predicted probabilities across ages (min..max)
        for each culture (useful for plotting trajectories)
    """
    import statsmodels.formula.api as smf
    import numpy as np
    import pandas as pd

    # Formula: main effects of age (centered), culture, their interaction, and controls
    # Use C(culture) so culture is treated as categorical
    formula = 'MajorityChoice ~ age_c * C(culture) + IsBoy + majority_first + religiousness + calworks'

    # Fit the logistic regression (maximum likelihood)
    # First fit a plain model to keep a reference to the raw results
    model_raw = smf.logit(formula=formula, data=df).fit(disp=False)

    # Attempt to obtain cluster-robust results. Some statsmodels versions support
    # get_robustcov_results on discrete model results; otherwise refit with cov_type.
    try:
        clustered_results = model_raw.get_robustcov_results(cov_type='cluster', groups=df['school'])
    except AttributeError:
        # Refit specifying cov_type='cluster' so that the returned results have cluster-robust cov
        clustered_results = smf.logit(formula=formula, data=df).fit(disp=False, cov_type='cluster', cov_kwds={'groups': df['school']})

    # Prepare predicted probabilities across ages for each culture for visualization
    age_min = int(df['age'].min())
    age_max = int(df['age'].max())
    ages = np.arange(age_min, age_max + 1)

    cultures = df['culture'].cat.categories
    mean_relig = df['religiousness'].mean()
    mean_calworks = df['calworks'].mean()
    # For majority_first use mode if available, otherwise mean then round to nearest 0/1
    if not df['majority_first'].mode().empty:
        mean_majority_first = int(df['majority_first'].mode().iloc[0])
    else:
        mean_majority_first = int(round(df['majority_first'].mean()))
    mean_IsBoy = df['IsBoy'].mean()

    pred_rows = []
    for c in cultures:
        for a in ages:
            age_c = a - df['age'].mean()
            age_c2 = age_c ** 2
            pred_rows.append({
                'age_c': age_c,
                'age_c2': age_c2,
                'culture': c,
                'IsBoy': mean_IsBoy,
                'majority_first': mean_majority_first,
                'religiousness': mean_relig,
                'calworks': mean_calworks
            })
    pred_df = pd.DataFrame(pred_rows)

    # Ensure culture dtype matches training data
    pred_df['culture'] = pred_df['culture'].astype(df['culture'].dtype)

    # Predicted probability using the fitted results (clustered_results has predict method)
    pred_df['pred_prob'] = clustered_results.predict(pred_df)

    results = {
        'model_raw': model_raw,
        'fit': clustered_results,
        'predicted_prob_grid': pred_df
    }

    return results