from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Example top-level read (kept from original file; transform accepts any df passed to it)
# df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/panda_nuts/anonymize_output/panda_nuts.csv')


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset into a dataframe ready for modeling. Produces the following columns used in the model:
      - SubjectID: integer identifier for the individual (from feature1)
      - Age: numeric age in years (from feature2)
      - Sex_Male: binary indicator 1 if sex is 'm' (from feature3)
      - HammerType: categorical hammer type (from feature4)
      - NutsOpened: integer count of nuts opened in session (from feature5)
      - SessionDuration_s: session duration in seconds (from feature6)
      - HelpBinary: binary indicator 1 if received help (from feature7)
      - Efficiency_npm: nuts opened per minute = NutsOpened / (SessionDuration_s / 60)
      - Efficiency_log: log1p(Efficiency_npm) (helpful diagnostic; not required for main model)
    """
    # Make a copy to avoid modifying the caller's data
    df = df.copy()

    # Rename raw columns to meaningful names
    rename_map = {
        'feature1': 'SubjectID',
        'feature2': 'Age',
        'feature3': 'Sex',
        'feature4': 'HammerType',
        'feature5': 'NutsOpened',
        'feature6': 'SessionDuration_s',
        'feature7': 'HelpRaw'
    }
    df = df.rename(columns=rename_map)

    # Ensure correct dtypes for numeric columns (keep as numpy dtypes)
    df['SubjectID'] = pd.to_numeric(df.get('SubjectID', pd.Series(dtype=float)), errors='coerce')
    df['Age'] = pd.to_numeric(df.get('Age', pd.Series(dtype=float)), errors='coerce')
    df['NutsOpened'] = pd.to_numeric(df.get('NutsOpened', pd.Series(dtype=float)), errors='coerce')
    df['SessionDuration_s'] = pd.to_numeric(df.get('SessionDuration_s', pd.Series(dtype=float)), errors='coerce')

    # Clean Sex -> intermediate mapping (use numpy float for NaN compatibility)
    df['Sex'] = df.get('Sex', pd.Series(dtype=str)).astype(str).str.strip().str.lower()
    sex_male_tmp = df['Sex'].map(lambda x: 1 if x == 'm' else (0 if x == 'f' else np.nan)).astype(float)
    df['Sex_Male'] = sex_male_tmp  # will be converted to integer after dropping missing rows

    # Clean HelpRaw -> intermediate HelpBinary
    df['HelpRaw'] = df.get('HelpRaw', pd.Series(dtype=str)).astype(str).str.strip().str.lower()
    help_tmp = df['HelpRaw'].map(lambda x: 1 if x in ['y', 'yes', 'true', '1'] else (0 if x in ['n', 'no', 'false', '0'] else np.nan)).astype(float)
    df['HelpBinary'] = help_tmp  # will be converted to integer after dropping missing rows

    # HammerType keep as categorical string-ish; normalize some missing/placeholder values
    if 'HammerType' in df.columns:
        # convert to string temporarily so we can catch literal 'nan' or 'None' strings, then set those to actual NA
        df['HammerType'] = df['HammerType'].astype(str).str.strip()
        df.loc[df['HammerType'].isin(['nan', 'none', 'None', '']), 'HammerType'] = pd.NA
    else:
        df['HammerType'] = pd.NA

    # Drop rows with essential missing values (use the required final columns)
    required_for_model = ['SubjectID', 'Age', 'NutsOpened', 'SessionDuration_s', 'Sex_Male', 'HelpBinary']
    df = df.dropna(subset=required_for_model)

    # Now that rows with missing essentials are removed, safely cast to numpy integer types
    # SubjectID should be integer identifier
    df['SubjectID'] = df['SubjectID'].astype(np.int64)

    # Sex_Male and HelpBinary should be binary integers 0/1
    df['Sex_Male'] = df['Sex_Male'].astype(np.int64)
    df['HelpBinary'] = df['HelpBinary'].astype(np.int64)

    # Ensure numeric columns are standard numpy dtypes
    df['Age'] = df['Age'].astype(np.float64)
    df['NutsOpened'] = df['NutsOpened'].astype(np.float64)
    df['SessionDuration_s'] = df['SessionDuration_s'].astype(np.float64)

    # Remove non-positive or extremely small session durations to avoid division by zero / invalid rates
    df = df[df['SessionDuration_s'] > 0]

    # Compute efficiency: nuts per minute
    df['Efficiency_npm'] = df['NutsOpened'] / (df['SessionDuration_s'] / 60.0)

    # Optionally create a log-transformed efficiency for diagnostics
    df['Efficiency_log'] = np.log1p(df['Efficiency_npm'])

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a regression model to estimate the effects of Age, Sex, and receiving help on nut-cracking efficiency.

    Model specification:
      - Outcome: Efficiency_npm (nuts per minute)
      - Predictors: Age (continuous), Sex_Male (binary), HelpBinary (binary)
      - Interactions: Age:HelpBinary and Sex_Male:HelpBinary to test whether help modifies the relationship of age/sex with efficiency
      - Controls: HammerType as a categorical fixed effect
      - Clustered standard errors at the SubjectID level to account for non-independence of sessions from the same individual

    Returns the fitted model results (statsmodels RegressionResultsWrapper).
    """
    # Ensure the columns needed for the model are present
    required_cols = ['Efficiency_npm', 'Age', 'Sex_Male', 'HelpBinary', 'HammerType', 'SubjectID']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Convert HammerType to a categorical factor for formula interface
    df = df.copy()
    df['HammerType'] = df['HammerType'].astype('category')

    # Build formula with interactions between HelpBinary and Age/Sex
    formula = 'Efficiency_npm ~ Age + Sex_Male + HelpBinary + Age:HelpBinary + Sex_Male:HelpBinary + C(HammerType)'

    # Fit OLS and use clustered robust standard errors by SubjectID
    ols_model = smf.ols(formula, data=df)
    results = ols_model.fit(cov_type='cluster', cov_kwds={'groups': df['SubjectID'].values})

    # Return the fitted results object (contains params, pvalues, summary, etc.)
    return results