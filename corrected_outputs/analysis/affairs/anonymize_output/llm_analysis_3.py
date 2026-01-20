from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/affairs/anonymize_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the Fair (Psychology Today) dataset for modeling the effect of having children on
    extramarital affair frequency.

    Outputs (columns used in the model):
      - AffairCount: numeric outcome (from feature2)
      - HasChildren: 0/1 indicator (from feature6)
      - Female: 0/1 indicator (from feature3)
      - Age_c: age (feature4) centered
      - YearsMarried_c: years married (feature5) centered
      - Religiosity, Education, Occupation, MarriageHappiness: numeric controls
    """

    # work on a copy
    df = df.copy()

    # Ensure affair count is numeric and drop rows missing the outcome or key IV
    df['AffairCount'] = pd.to_numeric(df['feature2'], errors='coerce')
    df['HasChildren'] = df['feature6'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})

    # Drop rows with missing outcome or missing children indicator
    df = df.dropna(subset=['AffairCount', 'HasChildren'])

    # Gender: map to Female indicator (female = 1, male = 0). Keep rows where gender is known.
    df['Female'] = df['feature3'].astype(str).str.strip().str.lower().map({'female': 1, 'male': 0})

    # Numeric controls: coerce to numeric, then drop rows with missing controls
    numeric_controls = ['feature4', 'feature5', 'feature7', 'feature8', 'feature9', 'feature10']
    for c in numeric_controls:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Rename and keep clean control columns
    df['Age'] = df['feature4']
    df['YearsMarried'] = df['feature5']
    df['Religiosity'] = df['feature7']
    df['Education'] = df['feature8']
    df['Occupation'] = df['feature9']
    df['MarriageHappiness'] = df['feature10']

    # Drop rows with any remaining missing values among the modeled columns
    required = ['AffairCount', 'HasChildren', 'Female', 'Age', 'YearsMarried', 'Religiosity', 'Education', 'Occupation', 'MarriageHappiness']
    df = df.dropna(subset=required)

    # Create centered versions of continuous covariates for interpretability
    df['Age_c'] = df['Age'] - df['Age'].mean()
    df['YearsMarried_c'] = df['YearsMarried'] - df['YearsMarried'].mean()

    # Keep only columns required for modeling (optional but clarifies final dataframe)
    model_cols = ['AffairCount', 'HasChildren', 'Female', 'Age_c', 'YearsMarried_c', 'Religiosity', 'Education', 'Occupation', 'MarriageHappiness']
    df = df[model_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a regression model to estimate the association between having children and
    extramarital affair frequency, controlling for measured covariates.

    Approach: use a Negative Binomial generalized linear model (GLM) because the
    outcome is a count-like variable with overdispersion relative to Poisson.

    Returns the fitted model results object.
    """

    import statsmodels.api as sm

    # Build the design matrix
    X = df[['HasChildren', 'Female', 'Age_c', 'YearsMarried_c', 'Religiosity', 'Education', 'Occupation', 'MarriageHappiness']].copy()
    X = sm.add_constant(X, has_constant='add')
    y = df['AffairCount']

    # Fit Negative Binomial GLM
    # Note: statsmodels' GLM NegativeBinomial uses the mean-variance relationship of NB.
    nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
    results = nb_model.fit()

    # It's useful to also provide robust SEs; user can inspect both.
    try:
        results_robust = results.get_robustcov_results(cov_type='HC3')
    except Exception:
        results_robust = None

    # Return a dict with the main results object and robust results (if available)
    return {
        'results': results,
        'results_robust': results_robust
    }


