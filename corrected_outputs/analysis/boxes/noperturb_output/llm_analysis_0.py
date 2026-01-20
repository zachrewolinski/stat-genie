from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/boxes/noperturb_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy to avoid side-effects
    df = df.copy()

    # Drop rows missing any key variables required for the analysis
    df = df.dropna(subset=['y', 'age', 'culture', 'gender', 'majority_first'])

    # Dependent variable: binary indicator for choosing the majority option (y == 2)
    df['MajorityChoice'] = (df['y'] == 2).astype(int)

    # Center age to aid interpretation of main effects and interactions
    df['age_c'] = df['age'].astype(float) - df['age'].astype(float).mean()

    # Culture: make a categorical label used in modeling (string/category form)
    # Keep the original numeric culture column if needed, but create culture_cat for use with C(...)
    df['culture_cat'] = df['culture'].astype('category').astype(str)

    # Gender: convert to binary indicator is_boy (1 = boy where original coding is 2 = boy)
    df['is_boy'] = (df['gender'] == 2).astype(int)

    # Ensure majority_first is integer 0/1
    df['majority_first'] = df['majority_first'].astype(int)

    # Return the transformed dataframe with all columns needed for modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    import statsmodels.formula.api as smf

    # Formula: main effect of centered age, main effect of culture (categorical), their interaction,
    # plus controls for gender and demonstration order.
    # Interaction age_c * C(culture_cat) allows the developmental slope to vary by culture.
    formula = 'MajorityChoice ~ age_c * C(culture_cat) + is_boy + majority_first'

    # Fit a logistic regression (binomial) model for the binary outcome MajorityChoice
    # Using statsmodels' logit (maximum likelihood). Suppress fit output with disp=False.
    model_fit = smf.logit(formula=formula, data=df).fit(disp=False)

    # Return the fitted model object (contains parameter estimates, standard errors, etc.)
    return model_fit


