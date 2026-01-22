from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/reading/anonymize_output/reading.csv')


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset into dataframe used for modeling. Creates:
      - ReaderView (0/1) from feature3
      - ReadingTime_ms, ReadingTime_s from feature5 (time on page minus scrolling)
      - WordCount from feature7
      - ReadingSpeed_wps = WordCount / ReadingTime_s
      - log_reading_speed = log(ReadingSpeed_wps)
      - Dyslexia (0/1) from feature17
      - DyslexiaSeverity from feature12
      - Age, Device, Education, Gender, IsNativeEnglish, Retake, Comprehension, ReadabilityFK, Page

    Performs basic cleaning: drops rows missing any required fields and filters implausible times.
    """

    df = df.copy()

    # --- Core variables ---
    # Reader View indicator (already 0/1 in feature3)
    df['ReaderView'] = pd.to_numeric(df['feature3'], errors='coerce')

    # Use the 'time on page minus scrolling' as the reading time (milliseconds)
    df['ReadingTime_ms'] = pd.to_numeric(df['feature5'], errors='coerce')
    # convert to seconds
    df['ReadingTime_s'] = df['ReadingTime_ms'] / 1000.0

    # Word count on page
    df['WordCount'] = pd.to_numeric(df['feature7'], errors='coerce')

    # Basic numeric controls
    df['Age'] = pd.to_numeric(df['feature10'], errors='coerce')
    df['Comprehension'] = pd.to_numeric(df['feature8'], errors='coerce')
    df['ReadabilityFK'] = pd.to_numeric(df['feature19'], errors='coerce')

    # Dyslexia indicators
    # feature17 is binary indicator (1 = dyslexia, 0 = no); ensure numeric
    df['Dyslexia'] = pd.to_numeric(df['feature17'], errors='coerce')
    # feature12 contains severity codes 0/1/2 (if present)
    df['DyslexiaSeverity'] = pd.to_numeric(df['feature12'], errors='coerce')

    # Categorical controls: Device, Education, Language, Page
    df['Device'] = df['feature11'].astype(str)
    df['Education'] = df['feature13'].astype(str)
    df['Language'] = df['feature15'].astype(str)
    df['Page'] = df['feature2'].astype(str)

    # Gender: map numeric codes to readable categories if possible
    # original coding: 0 - Male, 1 - Female, 2 - Other
    def _map_gender(x):
        try:
            x = int(x)
        except Exception:
            return 'Other'
        return {0: 'Male', 1: 'Female', 2: 'Other'}.get(x, 'Other')
    df['Gender'] = df['feature14'].apply(_map_gender)

    # Native English: feature18 contains 'Y'/'N'
    df['IsNativeEnglish'] = df['feature18'].map({'Y': 1, 'N': 0})

    # Retake indicator
    df['Retake'] = pd.to_numeric(df['feature16'], errors='coerce')

    # Drop rows missing key variables
    required = [
        'ReaderView', 'ReadingTime_s', 'WordCount', 'Dyslexia', 'Age',
        'Comprehension', 'ReadabilityFK', 'Device', 'Education', 'IsNativeEnglish', 'Page'
    ]
    df = df.dropna(subset=required)

    # Remove implausible or zero reading times/words
    df = df[df['ReadingTime_s'] > 0]
    df = df[df['WordCount'] > 0]

    # Compute reading speed (words per second) and log transform
    # Add small epsilon to avoid log(0) if numeric precision issues occur
    eps = 1e-8
    df['ReadingSpeed_wps'] = df['WordCount'] / df['ReadingTime_s']
    df['log_reading_speed'] = np.log(df['ReadingSpeed_wps'] + eps)

    # Optionally trim extreme outliers (e.g., extremely high speeds) to avoid undue influence
    # Here we trim the top 0.5% and bottom 0.5% of log_reading_speed
    low_q = df['log_reading_speed'].quantile(0.005)
    high_q = df['log_reading_speed'].quantile(0.995)
    df = df[(df['log_reading_speed'] >= low_q) & (df['log_reading_speed'] <= high_q)]

    # Keep only columns required for modeling plus helpful metadata
    keep_cols = [
        'ReaderView', 'Dyslexia', 'DyslexiaSeverity', 'Age', 'Gender', 'Retake',
        'Comprehension', 'ReadabilityFK', 'WordCount', 'Device', 'Education',
        'IsNativeEnglish', 'Page', 'ReadingTime_ms', 'ReadingTime_s',
        'ReadingSpeed_wps', 'log_reading_speed'
    ]

    df = df[keep_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS regression of log_reading_speed on ReaderView, Dyslexia and their interaction,
    controlling for demographic and trial-level covariates. Cluster standard errors by Page.

    Model:
      log_reading_speed ~ ReaderView * Dyslexia + Age + C(Gender) + Retake + Comprehension
                         + ReadabilityFK + WordCount + C(Device) + C(Education) + IsNativeEnglish

    Returns the fitted model object with cluster-robust covariance.
    """
    # Ensure categorical variables are treated as categories in the dataframe
    df = df.copy()
    df['Gender'] = df['Gender'].astype('category')
    df['Device'] = df['Device'].astype('category')
    df['Education'] = df['Education'].astype('category')
    df['Page'] = df['Page'].astype('category')

    # Define formula
    formula = (
        'log_reading_speed ~ ReaderView * Dyslexia + Age + C(Gender) + Retake + '
        'Comprehension + ReadabilityFK + WordCount + C(Device) + C(Education) + IsNativeEnglish'
    )

    # Fit OLS
    model_ols = smf.ols(formula=formula, data=df).fit()

    # Obtain cluster-robust standard errors clustered by Page.
    # Need to pass a groups array that is aligned to the rows actually used by the fitted model.
    # model_ols.model.data.row_labels contains the index labels of the original df rows used in the fit.
    try:
        row_idx = model_ols.model.data.row_labels
        # Select the Page values for those rows and convert to categorical codes for grouping
        groups = pd.Categorical(df.loc[row_idx, 'Page']).codes
    except Exception:
        # Fallback: if anything goes wrong, use the Page codes from the full df (should be aligned in normal cases)
        groups = pd.Categorical(df['Page']).codes

    clustered = model_ols.get_robustcov_results(cov_type='cluster', groups=groups)

    # Print a concise summary
    print(clustered.summary())

    # Return the robust results object so callers can inspect parameters, CIs, p-values, etc.
    return clustered