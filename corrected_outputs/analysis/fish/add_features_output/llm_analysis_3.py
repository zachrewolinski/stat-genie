from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/fish/add_features_output/fish.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataset for modeling fish caught per hour.

    Outputs (columns guaranteed present in returned df):
      - fish_caught: original count outcome (kept)
      - livebait: binary (0/1)
      - camper: binary (0/1)
      - persons: number of adults
      - child: number of children
      - total_people: persons + child
      - hours: hours spent in park (kept)
      - log_hours: natural log of hours (for offset)
      - fish_per_hour: fish_caught / hours (for descriptive summaries)
      - religiousness: kept as numeric control
      - year_centered: year - mean(year)
      - county: kept as categorical string/factor
    """
    df = df.copy()

    # Required columns for modeling - drop rows missing these
    required = ['fish_caught', 'hours', 'livebait', 'camper', 'persons', 'child', 'religiousness', 'year', 'county']
    # drop rows with NA in required columns
    df = df.dropna(subset=required)

    # Remove rows with non-positive or extremely small hours (cannot take log of 0)
    # Keep a minimal positive threshold to avoid numerical issues
    df = df[df['hours'] > 0]

    # Ensure binary columns are integers 0/1
    df['livebait'] = df['livebait'].astype(int)
    df['camper'] = df['camper'].astype(int)

    # Ensure numeric people counts
    df['persons'] = pd.to_numeric(df['persons'], errors='coerce')
    df['child'] = pd.to_numeric(df['child'], errors='coerce')
    df = df.dropna(subset=['persons', 'child'])

    # Derived variables
    df['total_people'] = df['persons'] + df['child']

    # Rate per hour (descriptive)
    df['fish_per_hour'] = df['fish_caught'] / df['hours']

    # log hours for offset (keep original hours as well)
    df['log_hours'] = np.log(df['hours'])

    # Center year to improve interpretability / numerical stability
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year'])
    df['year_centered'] = df['year'] - df['year'].mean()

    # Ensure county is treated as categorical (keep original string values)
    df['county'] = df['county'].astype(str)

    # Final: keep only columns needed downstream (but keep extras for inspection)
    keep_cols = ['fish_caught', 'livebait', 'camper', 'persons', 'child', 'total_people', 'hours', 'log_hours', 'fish_per_hour', 'religiousness', 'year_centered', 'county']
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a count regression for fish_caught using hours as exposure (offset).

    Workflow:
      1) Examine dispersion (variance/mean) of the count outcome to choose Poisson vs Negative Binomial.
      2) Fit GLM using formula with county as a categorical control: fish_caught ~ livebait + camper + total_people + religiousness + year_centered + C(county)
      3) Use log(hours) as the model offset so coefficients represent multiplicative effects on catch rate per hour.
      4) Return fitted model object, textual summary, and incident rate ratios (IRRs) with 95% CIs.
    """
    import statsmodels.formula.api as smf

    results = {}

    # Quick check: ensure necessary columns present
    required_cols = ['fish_caught', 'livebait', 'camper', 'total_people', 'religiousness', 'year_centered', 'county', 'log_hours', 'hours']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Transformed dataframe is missing required columns: {missing}")

    # Compute dispersion (variance / mean) to guide family choice
    mean_count = df['fish_caught'].mean()
    var_count = df['fish_caught'].var(ddof=1)
    dispersion = (var_count / mean_count) if mean_count > 0 else np.inf

    # Choose family: if overdispersed (dispersion > 1.5) prefer NegativeBinomial, else Poisson
    if dispersion > 1.5:
        family = sm.families.NegativeBinomial()
        chosen_family = 'NegativeBinomial'
    else:
        family = sm.families.Poisson()
        chosen_family = 'Poisson'

    formula = 'fish_caught ~ livebait + camper + total_people + religiousness + year_centered + C(county)'

    # Fit GLM with offset = log(hours)
    model = smf.glm(formula=formula, data=df, family=family, offset=df['log_hours'])
    fit = model.fit()

    # Summary text
    summary_text = fit.summary().as_text()

    # Compute Incident Rate Ratios (IRRs) and 95% CI by exponentiating coefficients
    params = fit.params
    conf = fit.conf_int()
    irr = np.exp(params)
    irr_lower = np.exp(conf[0])
    irr_upper = np.exp(conf[1])
    irr_df = pd.DataFrame({'IRR': irr, 'IRR_2.5%': irr_lower, 'IRR_97.5%': irr_upper})

    # Additional diagnostics: Pearson chi2 / df to inspect remaining overdispersion
    # For GLM families that provide resid_pearson
    try:
        pearson_chi2 = np.sum(fit.resid_pearson**2)
        rdf = fit.df_resid
        pearson_dispersion = pearson_chi2 / rdf if rdf > 0 else np.nan
    except Exception:
        pearson_chi2 = np.nan
        pearson_dispersion = np.nan

    results['chosen_family'] = chosen_family
    results['dispersion_raw'] = dispersion
    results['pearson_chi2'] = pearson_chi2
    results['pearson_dispersion'] = pearson_dispersion
    results['model'] = fit
    results['summary'] = summary_text
    results['irr'] = irr_df

    return results


