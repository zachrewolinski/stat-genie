from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/panda_nuts/anonymize_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset with columns feature1..feature7 into a cleaned dataframe containing
    the variables used in modeling.

    Output columns (kept/created):
      - ID (int)
      - Age (float)
      - Sex (categorical: 'M'/'F')
      - HammerType (categorical)
      - NutsOpened (float)
      - SessionDurationSec (float)
      - HelpReceived (int: 1 yes, 0 no)
      - Efficiency_per_min (float) = NutsOpened / (SessionDurationSec / 60)
      - LogEfficiency (float) = log1p(Efficiency_per_min)
    """
    df = df.copy()

    # 1. Rename raw columns to meaningful names
    df = df.rename(columns={
        'feature1': 'ID',
        'feature2': 'Age',
        'feature3': 'Sex',
        'feature4': 'HammerType',
        'feature5': 'NutsOpened',
        'feature6': 'SessionDurationSec',
        'feature7': 'HelpReceivedRaw'
    })

    # 2. Basic type conversion
    # Coerce numeric columns
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['NutsOpened'] = pd.to_numeric(df['NutsOpened'], errors='coerce')
    df['SessionDurationSec'] = pd.to_numeric(df['SessionDurationSec'], errors='coerce')

    # 3. Standardize Sex values to 'M'/'F' (if other encodings exist, map them to NA)
    df['Sex'] = df['Sex'].astype(str).str.strip().str.lower().map({'m': 'M', 'f': 'F'})

    # 4. Map help variable to binary 1/0. Expected values: 'y' or 'N' (from schema). Handle variations.
    df['HelpReceived'] = (
        df['HelpReceivedRaw'].astype(str).str.strip().str.lower().map({'y': 1, 'yes': 1, 'n': 0, 'no': 0})
    )

    # 5. Hammer type as string category
    df['HammerType'] = df['HammerType'].astype(str).str.strip()

    # 6. Compute efficiency: nuts per minute. Protect against zero/NA durations.
    # If SessionDurationSec <= 0 -> set to NaN
    df.loc[df['SessionDurationSec'] <= 0, 'SessionDurationSec'] = np.nan
    df['Efficiency_per_min'] = df['NutsOpened'] / (df['SessionDurationSec'] / 60.0)

    # 7. Log-transform the efficiency to reduce skew. Use log1p to handle zeros safely.
    df['LogEfficiency'] = np.log1p(df['Efficiency_per_min'])

    # 8. Drop rows with missing values in variables required for the model
    needed_cols = ['ID', 'Age', 'Sex', 'HelpReceived', 'HammerType', 'NutsOpened', 'SessionDurationSec', 'Efficiency_per_min', 'LogEfficiency']
    df = df.dropna(subset=needed_cols)

    # 9. Ensure ID is integer (for grouping in mixed model)
    # If ID is not integer coercible, keep as string/categorical; here we coerce to int when possible.
    try:
        df['ID'] = df['ID'].astype(int)
    except Exception:
        df['ID'] = df['ID'].astype(str)

    # 10. Keep only the columns we will use / inspect
    final_cols = ['ID', 'Age', 'Sex', 'HelpReceived', 'HammerType', 'NutsOpened', 'SessionDurationSec', 'Efficiency_per_min', 'LogEfficiency']
    df = df[final_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit a mixed-effects linear model predicting log-transformed efficiency (LogEfficiency)
    from Age, Sex, HelpReceived and their interactions, controlling for HammerType (fixed effect)
    and including a random intercept by ID.

    Model formula:
      LogEfficiency ~ Age + Sex + HelpReceived + Age:HelpReceived + Sex:HelpReceived + C(HammerType)

    Returns the fitted model result object (statsmodels MixedLMResults).
    """
    import statsmodels.formula.api as smf

    df = df.copy()

    # Ensure categorical variables are typed appropriately for the formula
    df['Sex'] = df['Sex'].astype('category')
    df['HammerType'] = df['HammerType'].astype('category')

    # Define formula including interactions between HelpReceived and Age / Sex
    formula = 'LogEfficiency ~ Age + Sex + HelpReceived + Age:HelpReceived + Sex:HelpReceived + C(HammerType)'

    # Fit a mixed-effects model with random intercept per ID to account for repeated measures
    # Use re_formula='1' to specify a random intercept only
    md = smf.mixedlm(formula, data=df, groups=df['ID'], re_formula='1')

    # Fit the model. Use default optimizer; if it fails in practice, try method='nm' or different options.
    mdf = md.fit(reml=False)

    # Print summary for immediate inspection; return the fitted model object for programmatic use
    print(mdf.summary())
    return mdf


