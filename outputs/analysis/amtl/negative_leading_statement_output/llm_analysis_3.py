from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/negative_leading_statement_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe suitable for binomial (AMTL) regression.

    Steps performed:
    - Drop rows missing the essential count data (num_amtl, sockets) or genus.
    - Ensure sockets > 0 (only meaningful observations).
    - Create amtl_events (num_amtl) and amtl_trials (sockets) columns.
    - Create amtl_prop = num_amtl / sockets (proportion) for modeling with binomial family and frequency weights.
    - Handle missing prob_male by imputing 0.5 (sex unknown) rather than dropping large numbers of rows.
    - Center age to age_c for interpretability (age - mean(age)).
    - Ensure genus and tooth_class are categorical with explicit reference levels (Pan as genus reference, Anterior as tooth_class reference).
    - Return dataframe with the added columns used by the model: amtl_events, amtl_trials, amtl_prop, age_c, and with categorical genus and tooth_class.
    """

    # Work on a copy
    df = df.copy()

    # Drop rows missing essential outcome or grouping variables
    df = df.dropna(subset=['num_amtl', 'sockets', 'genus', 'specimen', 'tooth_class', 'age'])

    # Keep only rows with positive number of observable sockets
    df = df[df['sockets'] > 0].copy()

    # Create explicit events/trials columns for binomial modeling
    df['amtl_events'] = df['num_amtl'].astype(int)
    df['amtl_trials'] = df['sockets'].astype(int)

    # Proportion (used as endog with freq_weights in GEE/GLM)
    df['amtl_prop'] = df['amtl_events'] / df['amtl_trials']

    # Impute missing prob_male to 0.5 (unknown sex) to retain observations rather than dropping them.
    if 'prob_male' in df.columns:
        df['prob_male'] = df['prob_male'].fillna(0.5)
    else:
        # If the column is missing entirely, create a neutral value
        df['prob_male'] = 0.5

    # Center age to improve interpretability and numerical stability
    df['age_c'] = df['age'] - df['age'].mean()

    # Set categorical variables with chosen reference levels.
    # Reference for genus: 'Pan' (chimpanzees) so that coefficients compare other genera to Pan.
    # Ensure the category names present in the data are preserved; if some expected levels are absent, categories will contain only present levels.
    genus_categories = ['Pan', 'Pongo', 'Papio', 'Homo sapiens']
    # Keep only categories that actually appear, but put Pan first if present
    present = [g for g in genus_categories if g in df['genus'].unique()]
    if 'Pan' in present:
        df['genus'] = pd.Categorical(df['genus'], categories=present)
    else:
        # If Pan not present in this subset, use the sorted unique values but try to keep a stable ordering
        df['genus'] = pd.Categorical(df['genus'])

    # For tooth_class, make 'Anterior' the reference
    tooth_order = ['Anterior', 'Premolar', 'Posterior']
    present_tc = [t for t in tooth_order if t in df['tooth_class'].unique()]
    if present_tc:
        df['tooth_class'] = pd.Categorical(df['tooth_class'], categories=present_tc)
    else:
        df['tooth_class'] = pd.Categorical(df['tooth_class'])

    # Final safety: drop any rows where proportion is NaN or outside [0,1]
    df = df[df['amtl_prop'].notnull()]
    df = df[(df['amtl_prop'] >= 0) & (df['amtl_prop'] <= 1)]

    # Return transformed dataframe used in modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a binomial GEE to estimate whether Homo sapiens has higher AMTL than non-human primate genera
    after accounting for age, sex (prob_male), and tooth class. Uses specimen as the clustering/grouping variable
    (exchangeable working correlation). Uses amtl_prop as the response and amtl_trials as freq_weights for binomial trials.

    Returns a dictionary containing:
    - 'gee_result': the fitted GEE result object
    - 'contrasts': a pandas DataFrame with pairwise contrasts of Homo sapiens vs each non-human genus (log-OR, OR, 95% CI, z, p)

    Notes:
    - The model formula uses the first category of 'genus' as the reference. The transform function attempts to set 'Pan' as that reference.
    """
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from scipy import stats

    # Ensure required columns are present
    required = ['amtl_prop', 'amtl_trials', 'genus', 'age_c', 'prob_male', 'tooth_class', 'specimen']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe passed to model().")

    # Build formula. C(genus) and C(tooth_class) will use the categorical ordering defined during transform().
    formula = 'amtl_prop ~ C(genus) + age_c + prob_male + C(tooth_class)'

    # Fit GEE with binomial family and exchangeable correlation within specimen clusters.
    # Use freq_weights = amtl_trials so the binomial denominator is respected.
    model_gee = sm.GEE.from_formula(formula,
                                   groups='specimen',
                                   data=df,
                                   family=sm.families.Binomial(),
                                   cov_struct=sm.cov_struct.Exchangeable(),
                                   freq_weights=df['amtl_trials'])

    gee_result = model_gee.fit()

    # Prepare pairwise contrasts comparing Homo sapiens to each non-human genus.
    # Identify parameter names and covariance matrix
    params = gee_result.params
    cov = gee_result.cov_params()
    param_names = params.index.tolist()

    # Helper to build contrast vector for comparing Homo sapiens vs a given genus X.
    def contrast_vector_vs_genus(genus_x, genus_human='Homo sapiens'):
        # Create zero vector
        vec = np.zeros(len(param_names))
        # If comparing Homo sapiens to Pan (reference), the contrast is simply the Homo coefficient (if present).
        # For general case, contrast = coef(Homo) - coef(genus_x). If genus_x is the reference (Pan), coef for Pan is embedded in intercept.
        # So we look for parameter names of the form 'C(genus)[T.<level>]'
        name_h = f"C(genus)[T.{genus_human}]"
        name_x = f"C(genus)[T.{genus_x}]"
        if name_h in param_names:
            vec[param_names.index(name_h)] = 1.0
        else:
            # If Homo coefficient not present, it may be that Homo is the reference (unlikely given transform), so handle gracefully
            raise ValueError(f"Parameter {name_h} not found in fitted parameters: {param_names}")

        if name_x in param_names:
            vec[param_names.index(name_x)] = -1.0
        else:
            # If genus_x is the reference (e.g., Pan) there will be no explicit parameter; that's fine -- contrast already set
            pass

        return vec

    # Determine which non-human genera are present in data (excluding Homo sapiens)
    present_genera = [g for g in df['genus'].cat.categories if pd.notna(g) and g != 'Homo sapiens']

    contrasts_list = []
    for g in present_genera:
        vec = contrast_vector_vs_genus(g)
        est = float(np.dot(vec, params.values))
        var = float(np.dot(vec, np.dot(cov.values, vec)))
        se = np.sqrt(var) if var > 0 else np.nan
        z = est / se if se and not np.isnan(se) else np.nan
        p = 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
        or_est = np.exp(est)
        ci_low = np.exp(est - 1.96 * se)
        ci_upp = np.exp(est + 1.96 * se)

        contrasts_list.append({
            'contrast': f'Homo sapiens vs {g}',
            'log_odds_ratio': est,
            'OR': or_est,
            'OR_95CI_low': ci_low,
            'OR_95CI_high': ci_upp,
            'z': z,
            'p_two_tailed': p
        })

    contrasts_df = pd.DataFrame(contrasts_list)

    # Assemble results
    results = {
        'gee_result': gee_result,
        'contrasts': contrasts_df
    }

    return results


