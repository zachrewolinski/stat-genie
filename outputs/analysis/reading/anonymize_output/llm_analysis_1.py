from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# If running as a script, the caller may replace this path or not use it.
# Keep the original read for compatibility with the provided context.
try:
    df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/reading/anonymize_output/reading.csv')
except Exception:
    df = pd.DataFrame()


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Rename raw feature columns to descriptive names used in the model
    rename_map = {
        'feature1': 'ParticipantID',
        'feature2': 'PageID',
        'feature3': 'ReaderView',        # 0/1
        'feature4': 'PageTime_ms',
        'feature5': 'ReadingTime_ms',    # non-scrolling time (ms)
        'feature6': 'ScrollingTime_ms',
        'feature7': 'Words',
        'feature8': 'Comprehension',
        'feature9': 'ImageWidth',
        'feature10': 'Age',
        'feature11': 'Device',
        'feature12': 'DyslexiaSeverity', # 0/1/2
        'feature13': 'Education',
        'feature14': 'Gender',
        'feature15': 'Language',
        'feature16': 'IsRetake',
        'feature17': 'IsDyslexic',     # 0/1
        'feature18': 'NativeEnglish',   # 'Y'/'N'
        'feature19': 'FK_readability',
        'feature20': 'feature20'
    }
    df = df.rename(columns=rename_map)

    # Keep only rows with the minimum required info (only dropna for columns that exist)
    required_cols = ['ReaderView', 'ReadingTime_ms', 'Words', 'IsDyslexic', 'ParticipantID']
    df = df.dropna(subset=[c for c in required_cols if c in df.columns])

    # Convert data types (guarded conversions)
    if 'ReaderView' in df.columns:
        try:
            df['ReaderView'] = df['ReaderView'].astype(int)
        except Exception:
            # attempt more permissive conversion
            df['ReaderView'] = pd.to_numeric(df['ReaderView'], errors='coerce').fillna(0).astype(int)

    if 'IsDyslexic' in df.columns:
        try:
            df['IsDyslexic'] = df['IsDyslexic'].astype(int)
        except Exception:
            df['IsDyslexic'] = pd.to_numeric(df['IsDyslexic'], errors='coerce').fillna(0).astype(int)

    # Basic derived columns
    if 'ReadingTime_ms' in df.columns:
        df['ReadingTime_sec'] = pd.to_numeric(df['ReadingTime_ms'], errors='coerce') / 1000.0
    else:
        df['ReadingTime_sec'] = np.nan

    # Remove implausible / zero reading times
    # Only filter if the column exists and is numeric
    if 'ReadingTime_sec' in df.columns:
        df = df[df['ReadingTime_sec'] > 0.2]

    # Compute reading speed: words per minute
    if 'Words' in df.columns:
        df['Words'] = pd.to_numeric(df['Words'], errors='coerce')
        df.loc[df['Words'] <= 0, 'Words'] = np.nan
        df = df.dropna(subset=['Words'])
        # Avoid division by zero or NaN in ReadingTime_sec
        df['ReadingSpeedWPM'] = df['Words'] * 60.0 / df['ReadingTime_sec']
    else:
        df['ReadingSpeedWPM'] = np.nan

    # Filter out trials marked as retakes to keep fresh reading trials (optional but typical)
    if 'IsRetake' in df.columns:
        try:
            df['IsRetake'] = pd.to_numeric(df['IsRetake'], errors='coerce').fillna(0).astype(int)
            df = df[df['IsRetake'] == 0]
        except Exception:
            # If conversion fails, skip filtering
            pass

    # Remove extremely implausible reading speeds using robust percentile trimming (0.5th - 99.5th)
    if 'ReadingSpeedWPM' in df.columns and df['ReadingSpeedWPM'].notnull().sum() >= 10:
        low, high = np.percentile(df['ReadingSpeedWPM'].dropna(), [0.5, 99.5])
        df = df[(df['ReadingSpeedWPM'] >= low) & (df['ReadingSpeedWPM'] <= high)]

    # Create binary native-English column: 1 for 'Y', 0 for 'N' or missing/others
    if 'NativeEnglish' in df.columns:
        df['IsNativeEnglish'] = df['NativeEnglish'].astype(str).str.upper().map({'Y': 1, 'N': 0})
        df['IsNativeEnglish'] = df['IsNativeEnglish'].fillna(0).astype(int)
    else:
        df['IsNativeEnglish'] = 0

    # Ensure categorical columns are of type string (so formula C() will treat them as categories)
    for col in ['Device', 'Education', 'Language', 'ParticipantID', 'PageID']:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Ensure final dataframe contains all required conceptual columns (create if missing)
    conceptual_cols_defaults = {
        'ParticipantID': None,
        'ReaderView': np.nan,
        'ReadingSpeedWPM': np.nan,
        'IsDyslexic': np.nan,
        'Age': np.nan,
        'Words': np.nan,
        'Comprehension': np.nan,
        'Device': np.nan,
        'IsNativeEnglish': 0,
        'Education': np.nan,
        'Language': np.nan
    }
    for col, default in conceptual_cols_defaults.items():
        if col not in df.columns:
            df[col] = default

    # For consistent behavior, make ParticipantID strings (required for grouping)
    if 'ParticipantID' in df.columns:
        df['ParticipantID'] = df['ParticipantID'].astype(str)

    # Keep and return only the columns needed for modelling and interpretation
    keep_cols = [
        'ParticipantID', 'ReaderView', 'ReadingSpeedWPM', 'IsDyslexic',
        'Age', 'Words', 'Comprehension', 'Device', 'IsNativeEnglish',
        'Education', 'Language'
    ]
    # Subset to existing columns in df (they should all exist after defaults)
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a linear mixed-effects model predicting ReadingSpeedWPM.
    The focal test is the interaction between ReaderView and IsDyslexic.

    Model formula:
      ReadingSpeedWPM ~ ReaderView * IsDyslexic + Age + Comprehension + Words + C(Device) + C(Education) + C(Language) + IsNativeEnglish

    Random effects: random intercept for ParticipantID to account for repeated measures.
    """
    # Ensure required columns exist
    required = ['ReadingSpeedWPM', 'ReaderView', 'IsDyslexic', 'ParticipantID']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Required column {c} not found in dataframe")

    # Build formula. Use C(...) to include categorical covariates if available.
    formula_terms = ['ReaderView * IsDyslexic', 'Age', 'Comprehension', 'Words', 'IsNativeEnglish']
    if 'Device' in df.columns:
        formula_terms.append('C(Device)')
    if 'Education' in df.columns:
        formula_terms.append('C(Education)')
    if 'Language' in df.columns:
        formula_terms.append('C(Language)')

    formula = 'ReadingSpeedWPM ~ ' + ' + '.join(formula_terms)

    # Before fitting, drop rows with NA in any variables used by the model
    vars_needed = ['ReadingSpeedWPM', 'ReaderView', 'IsDyslexic', 'Age', 'Comprehension', 'Words', 'IsNativeEnglish', 'ParticipantID']
    if 'Device' in df.columns:
        vars_needed.append('Device')
    if 'Education' in df.columns:
        vars_needed.append('Education')
    if 'Language' in df.columns:
        vars_needed.append('Language')
    # Keep only the vars that actually exist in df to avoid errors
    vars_needed = [v for v in vars_needed if v in df.columns]
    df_clean = df.dropna(subset=vars_needed).reset_index(drop=True)

    if df_clean.shape[0] == 0:
        raise ValueError("No data available after dropping missing values required for the model.")

    # Import mixedlm from statsmodels.formula.api
    from statsmodels.formula.api import mixedlm

    # Ensure ParticipantID is a suitable grouping variable (string/categorical)
    if 'ParticipantID' in df_clean.columns:
        df_clean['ParticipantID'] = df_clean['ParticipantID'].astype(str)

    # Fit mixed effects model with random intercept for ParticipantID by specifying the group vector.
    # Pass the actual grouping series so statsmodels can correctly assign observations to groups.
    md = mixedlm(formula, df_clean, groups=df_clean['ParticipantID'])
    mdf = md.fit(reml=False)

    # Print and return the fitted model object. The user can inspect mdf.summary()
    print(mdf.summary())
    return mdf