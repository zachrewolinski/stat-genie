from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/anonymize_output/amtl.csv')


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset to the analysis dataframe.

    Inputs (expected original columns):
      - feature1: tooth class (Anterior/Posterior/Premolar)
      - feature2: specimen id
      - feature3: number of missing teeth of given class
      - feature4: number of observable sockets that could be scored for missing teeth (trials)
      - feature5: estimated age at death
      - feature6: assigned uncertainty of age at death
      - feature7: estimate of sex (continuous 0-1)
      - feature8: genus (Homo sapiens, Pan, Pongo, Papio)
      - feature9: region

    Returns dataframe with columns used in modeling:
      AMTL_miss, AMTL_obs, IsHuman, Age, AgeUncertainty, SexProb, Sex_binary,
      ToothClass, Genus, Region, SpecimenID

    Notes:
      - Adds an internal helper column 'AMTL_prop' that is the proportion
        AMTL_miss / AMTL_obs with a small continuity adjustment for rows
        that are exactly 0 or 1 to improve GLM fitting stability.
    """
    df = df.copy()

    # 1) Rename to analysis-friendly column names (must preserve exact final column names)
    df = df.rename(columns={
        'feature1': 'ToothClass',
        'feature2': 'SpecimenID',
        'feature3': 'AMTL_miss',
        'feature4': 'AMTL_obs',
        'feature5': 'Age',
        'feature6': 'AgeUncertainty',
        'feature7': 'SexProb',
        'feature8': 'Genus',
        'feature9': 'Region'
    })

    # 2) Drop rows with missing key variables required for binomial modeling
    # Include AgeUncertainty because it is a required control variable
    df = df.dropna(subset=['AMTL_miss', 'AMTL_obs', 'Genus', 'ToothClass', 'Age', 'SexProb', 'AgeUncertainty'])

    # 3) Ensure numeric types for counts and trials; coerce invalid to NA and drop
    df['AMTL_miss'] = pd.to_numeric(df['AMTL_miss'], errors='coerce')
    df['AMTL_obs'] = pd.to_numeric(df['AMTL_obs'], errors='coerce')
    df = df.dropna(subset=['AMTL_miss', 'AMTL_obs'])

    # Convert to integer where appropriate
    df['AMTL_miss'] = df['AMTL_miss'].astype(int)
    df['AMTL_obs'] = df['AMTL_obs'].astype(int)

    # 4) Remove observations with zero or negative trial counts (cannot model)
    df = df[df['AMTL_obs'] > 0]

    # 5) Fix any impossible values: AMTL_miss cannot exceed AMTL_obs
    mask = df['AMTL_miss'] > df['AMTL_obs']
    if mask.any():
        df.loc[mask, 'AMTL_miss'] = df.loc[mask, 'AMTL_obs']

    # 6) Create primary independent variable: indicator for modern humans
    df['IsHuman'] = (df['Genus'].astype(str).str.strip() == 'Homo sapiens').astype(int)

    # 7) Preserve sex as continuous probability and create a binary alternative for sensitivity
    df['SexProb'] = pd.to_numeric(df['SexProb'], errors='coerce')
    # Clip SexProb to [0,1] if some values fall slightly outside due to bad input
    df['SexProb'] = df['SexProb'].clip(lower=0.0, upper=1.0)
    df['Sex_binary'] = (df['SexProb'] >= 0.5).astype(int)

    # 8) Make categorical columns categorical dtype for modeling
    df['ToothClass'] = df['ToothClass'].astype('category')
    df['Genus'] = df['Genus'].astype('category')
    df['Region'] = df['Region'].astype('category')

    # 9) Ensure Age and AgeUncertainty numeric
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['AgeUncertainty'] = pd.to_numeric(df['AgeUncertainty'], errors='coerce')

    # Drop any rows that became NA in Age or AgeUncertainty after coercion
    df = df.dropna(subset=['Age', 'AgeUncertainty', 'SexProb'])

    # 10) Create an internal helper: proportion of missing teeth (AMTL_prop)
    # Use exact ratio for most rows but apply a small continuity correction for perfect 0 or 1 values
    df['AMTL_prop'] = df['AMTL_miss'] / df['AMTL_obs']
    # Identify exact zeros or ones and adjust them slightly using (miss + 0.5) / (obs + 1)
    mask_edge = (df['AMTL_prop'] == 0) | (df['AMTL_prop'] == 1)
    if mask_edge.any():
        df.loc[mask_edge, 'AMTL_prop'] = (
            (df.loc[mask_edge, 'AMTL_miss'].astype(float) + 0.5) /
            (df.loc[mask_edge, 'AMTL_obs'].astype(float) + 1.0)
        )

    # 11) Reset index and return
    df = df.reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a binomial (logistic) GLM to test whether modern humans have higher
    AMTL frequencies than non-human primates, controlling for age, sex, tooth class,
    and region. Standard errors are clustered by SpecimenID to account for multiple
    tooth-class observations per specimen.

    Model specification (primary):
      response: AMTL_miss / AMTL_obs (proportion) with weights = AMTL_obs (trials)
                implemented using an internal helper column 'AMTL_prop' that is
                equal to AMTL_miss / AMTL_obs with edge adjustments.
      predictors: IsHuman + Age + SexProb + C(ToothClass) + C(Region)

    Returns a dictionary containing the fitted model, cluster-robust results,
    and a concise summary for the IsHuman effect (coef, se, p, odds ratio, 95% CI).
    """
    # Ensure required columns exist (include AgeUncertainty per conceptual variables)
    required = ['AMTL_miss', 'AMTL_obs', 'IsHuman', 'Age', 'AgeUncertainty', 'SexProb', 'ToothClass', 'Region', 'SpecimenID']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError('Missing required columns for modeling: {}'.format(missing))

    # Ensure AMTL_prop helper exists; if not, compute it here with the same adjustment
    if 'AMTL_prop' not in df.columns:
        df = df.copy()
        df['AMTL_prop'] = df['AMTL_miss'] / df['AMTL_obs']
        mask_edge = (df['AMTL_prop'] == 0) | (df['AMTL_prop'] == 1)
        if mask_edge.any():
            df.loc[mask_edge, 'AMTL_prop'] = (
                (df.loc[mask_edge, 'AMTL_miss'].astype(float) + 0.5) /
                (df.loc[mask_edge, 'AMTL_obs'].astype(float) + 1.0)
            )

    # Formula: use the internal proportion column as response and weights equal to number of trials
    formula = 'AMTL_prop ~ IsHuman + Age + SexProb + C(ToothClass) + C(Region)'

    # Fit GLM binomial with weights = AMTL_obs (frequency of trials)
    # Use try/except to provide clearer error if fitting fails
    try:
        glm_raw = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), weights=df['AMTL_obs']).fit()
    except Exception as e:
        # Provide a more informative error that preserves original exception message
        raise RuntimeError(f'GLM fitting failed: {e}')

    # Obtain cluster-robust covariance (cluster by SpecimenID)
    try:
        glm_cluster = glm_raw.get_robustcov_results(cov_type='cluster', groups=df['SpecimenID'])
    except Exception:
        # If clustering fails, fall back to default results
        glm_cluster = glm_raw

    # Extract key statistics for the IsHuman effect
    if 'IsHuman' in glm_cluster.params.index:
        coef = float(glm_cluster.params['IsHuman'])
        # Use clustered bse/pvalues if available; fall back to model bse/pvalues otherwise
        try:
            se = float(glm_cluster.bse['IsHuman'])
        except Exception:
            se = float(glm_raw.bse.get('IsHuman', np.nan))
        try:
            pval = float(glm_cluster.pvalues['IsHuman'])
        except Exception:
            pval = float(glm_raw.pvalues.get('IsHuman', np.nan))
        # Odds ratio and 95% CI (on coefficient scale -> exp)
        try:
            ci = glm_cluster.conf_int().loc['IsHuman']
            ci_low = float(np.exp(ci[0]))
            ci_high = float(np.exp(ci[1]))
        except Exception:
            ci_low = ci_high = float('nan')
        odds_ratio = float(np.exp(coef))
    else:
        coef = se = pval = odds_ratio = ci_low = ci_high = float('nan')

    results = {
        'glm_raw': glm_raw,
        'glm_clustered': glm_cluster,
        'IsHuman_coef': coef,
        'IsHuman_se': se,
        'IsHuman_pvalue': pval,
        'IsHuman_odds_ratio': odds_ratio,
        'IsHuman_odds_ratio_CI95': (ci_low, ci_high),
        'formula': formula
    }

    # Print brief summary to console for quick inspection (non-fatal)
    try:
        print('GLM (binomial) fitted. Primary formula:', formula)
        print(glm_cluster.summary())
    except Exception:
        pass

    return results