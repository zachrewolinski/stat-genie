from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/noperturb_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for modeling.

    Steps:
    - Drop rows with missing values in variables needed for the model.
    - Create binary outcome y_majority (1 if y==2, else 0).
    - Center age (age_c) and create quadratic term age2 to capture nonlinearity.
    - Create is_boy indicator from gender (1=girl, 2=boy in raw data).
    - Ensure majority_first is binary (0/1).
    - Create culture dummy variables (culture_2 ... culture_8) with culture_1 as reference.
    - Create interaction terms between centered age and each culture dummy.
    - Add an Intercept column for explicit design matrix use.

    The returned dataframe will contain at minimum the columns referenced in the conceptual variables and model code.
    """
    # Required columns for modeling
    required_cols = ['y', 'age', 'culture', 'gender', 'majority_first']
    df = df.copy()

    # Drop rows with missing values in any required column
    df = df.dropna(subset=required_cols)

    # Create binary dependent variable: 1 if majority option selected (y == 2), else 0
    df['y_majority'] = (df['y'] == 2).astype(int)

    # Center age and create quadratic term
    df['age_c'] = df['age'] - df['age'].mean()
    df['age2'] = df['age_c'] ** 2

    # Convert gender to is_boy indicator: gender==2 -> 1, else 0
    df['is_boy'] = (df['gender'] == 2).astype(int)

    # Ensure majority_first is 0/1 integer
    df['majority_first'] = df['majority_first'].astype(int)

    # Create culture dummies; drop_first=True uses culture_1 as reference
    # Cast culture to string to get deterministic column names like 'culture_2'
    culture_dummies = pd.get_dummies(df['culture'].astype(int).astype(str), prefix='culture', drop_first=True)
    # Ensure consistent ordering of dummy columns (culture_2 ... culture_8 if present)
    culture_dummies = culture_dummies.reindex(sorted(culture_dummies.columns, key=lambda x: int(x.split('_')[1])), axis=1).fillna(0).astype(int)

    # Attach culture dummies to df
    for col in culture_dummies.columns:
        df[col] = culture_dummies[col].values

    # Create interactions between centered age and each culture dummy
    interaction_cols = []
    for col in culture_dummies.columns:
        inter_name = 'age_c_x_' + col
        df[inter_name] = df['age_c'] * df[col]
        interaction_cols.append(inter_name)

    # Add explicit intercept column for modeling
    df['Intercept'] = 1.0

    # Optionally, build a list of model columns to keep for downstream modeling
    model_cols = ['Intercept', 'age_c', 'age2', 'is_boy', 'majority_first'] + list(culture_dummies.columns) + interaction_cols
    # We keep these columns (plus the outcome) in the returned dataframe to make subsequent modeling explicit
    cols_to_return = ['y_majority'] + model_cols + ['y', 'age', 'culture', 'gender', 'majority_first']

    # Return df with at least the columns we need (keeping other columns as well is fine)
    return df[cols_to_return].copy()


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial (logistic) regression predicting the probability of choosing the majority option.

    Model specification (GLM, binomial logit link):
      y_majority ~ age_c + age2 + is_boy + majority_first + culture dummies + (age_c x culture dummies)

    Interpretation focus: the age_c x culture interaction terms test whether the developmental slope for majority preference differs across cultural sites.

    Returns the fitted statsmodels GLM results object.
    """
    # Ensure necessary model columns exist
    # Base columns
    base_cols = ['Intercept', 'age_c', 'age2', 'is_boy', 'majority_first']
    # Culture dummy columns expected (culture_2 ... culture_8). Keep only those present in df.
    culture_cols = [c for c in df.columns if c.startswith('culture_')]
    # Interaction columns: age_c_x_culture_*
    interaction_cols = [c for c in df.columns if c.startswith('age_c_x_culture_')]

    model_cols = base_cols + sorted(culture_cols) + sorted(interaction_cols)

    # Subset to rows without missing values in model columns or the outcome
    model_df = df.dropna(subset=['y_majority'] + model_cols)

    # Design matrix X and outcome y
    X = model_df[model_cols].astype(float)
    y = model_df['y_majority'].astype(int)

    # Fit GLM with binomial family (logit link)
    model = sm.GLM(y, X, family=sm.families.Binomial())
    results = model.fit()

    # Optionally compute clustered robust SEs by culture (uncomment if desired):
    # results_robust = results.get_robustcov_results(cov_type='cluster', groups=model_df['culture'])
    # return results_robust

    return results


