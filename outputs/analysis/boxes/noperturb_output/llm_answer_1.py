def extract_final_answer(model_output):
    """
    Extracts the effect of age (age_z) on choosing the majority option (category 1 vs reference 0)
    across all cultural sites, using the fitted statsmodels MNLogit result object.

    Returns a dictionary with:
      - "object": a pandas.DataFrame summarizing, for each culture (1..K), the estimated
                  linear effect of age_z on the log-odds of choosing the majority (vs unchosen),
                  its standard error, z-value, two-sided p-value, and 95% CI.
      - "description": brief explanation of what the numbers mean.

    The function handles the common statsmodels MNLogit output layout:
      - model_output.params is expected to be a DataFrame with one row per non-reference outcome
        (e.g., indices 1 and 2) and columns = exog names.
      - model_output.cov_params() is expected to be the full covariance matrix for the
        flattened parameter vector; the function slices out the block for the outcome=majority.
    """
    import numpy as np
    import pandas as pd
    import math

    # Try to import a normal CDF for p-value calculation; fall back to erf-based
    try:
        from scipy import stats
        norm_cdf = stats.norm.cdf
    except Exception:
        def norm_cdf(x):
            # using error function: cdf = 0.5*(1 + erf(x/sqrt(2)))
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    # 1) Access parameter table
    params = model_output.params  # likely a DataFrame with index = outcomes (1,2) and cols = exog names
    exog_names = list(model_output.model.exog_names)  # list of exogenous variable names in order
    k_exog = len(exog_names)

    # Ensure params is a DataFrame in expected shape
    if isinstance(params, pd.DataFrame):
        outcome_indices = list(params.index)
    else:
        # if it's an ndarray (J-1, k), create synthetic indices 0..J-2
        params = pd.DataFrame(params)
        outcome_indices = list(params.index)

    # We want the majority outcome which is coded as 1 in the task
    # Find which row corresponds to outcome '1' (could be int or string)
    target_outcome = None
    for idx in outcome_indices:
        try:
            if int(idx) == 1:
                target_outcome = idx
                break
        except Exception:
            # idx may be string
            if str(idx) == '1':
                target_outcome = idx
                break
    if target_outcome is None:
        # If not found, assume the first non-reference outcome corresponds to majority (best effort)
        target_outcome = outcome_indices[0]

    # Extract the coefficient vector (Series) for the majority outcome
    beta_row = params.loc[target_outcome]
    # If params had shape (k,) as a Series, make it consistent
    beta_row = pd.Series(beta_row, index=exog_names)

    # 2) Extract the covariance block corresponding to the majority outcome parameters
    cov_full = model_output.cov_params()  # DataFrame or ndarray of shape ((J-1)*k, (J-1)*k)
    # Determine the block position of target_outcome among outcome_indices
    pos = outcome_indices.index(target_outcome)
    # Compute row/col indices for the slice
    start = pos * k_exog
    end = start + k_exog

    # Convert cov_full to numpy array for safe slicing (it might be DataFrame with non-numeric index)
    cov_full_arr = np.asarray(cov_full)
    cov_block = cov_full_arr[start:end, start:end]  # k x k covariance for this outcome

    # 3) Identify culture dummy names and interaction names from exog_names
    # The code that fitted the model used names like 'culture_2', 'age_z_x_culture_2', etc.
    culture_dummy_names = [n for n in exog_names if n.startswith('culture_')]
    # Try to infer available culture numeric IDs from dummy names
    culture_ids = []
    for name in culture_dummy_names:
        # name like 'culture_2' -> id 2
        try:
            suffix = name.split('_')[-1]
            culture_ids.append(int(suffix))
        except Exception:
            pass
    # Determine maximum culture id (if culture ids found). We assume cultures are 1..K and culture_1 was dropped.
    if culture_ids:
        max_cid = max(culture_ids + [1])  # include 1 for reference
        K = max_cid
    else:
        # fallback: assume 1..8 as described in the task
        K = 8

    # Precompute index positions of relevant exog columns
    exog_index_map = {name: idx for idx, name in enumerate(exog_names)}
    if 'age_z' not in exog_index_map:
        raise ValueError("age_z not found among model exogenous variable names.")

    idx_age = exog_index_map['age_z']

    # Prepare results list
    rows = []
    for cid in range(1, K + 1):
        # construct contrast vector c of length k_exog
        c = np.zeros(k_exog, dtype=float)
        c[idx_age] = 1.0
        if cid != 1:
            inter_name = f'age_z_x_culture_{cid}'
            if inter_name in exog_index_map:
                c[exog_index_map[inter_name]] = 1.0
            else:
                # If the exact interaction name not present, try to find any interaction that endswith the culture id
                matches = [n for n in exog_names if n.endswith(f'_{cid}') and n.startswith('age_z_x_')]
                if matches:
                    c[exog_index_map[matches[0]]] = 1.0
                # else leave as just the main age effect (assumes no interaction for this culture)
        # Estimate, SE, z, p, CI
        est = float(np.dot(c, beta_row.values))
        var = float(np.dot(c, cov_block.dot(c)))
        se = math.sqrt(var) if var >= 0 else float('nan')
        z = est / se if se and not math.isnan(se) else float('nan')
        # two-sided p-value
        p = 2.0 * (1.0 - norm_cdf(abs(z))) if not math.isnan(z) else float('nan')
        ci_low = est - 1.96 * se if not math.isnan(se) else float('nan')
        ci_high = est + 1.96 * se if not math.isnan(se) else float('nan')

        rows.append({
            'culture_id': cid,
            'age_effect_log_odds_majority_vs_ref': est,
            'se': se,
            'z': z,
            'p_two_sided': p,
            'ci95_lower': ci_low,
            'ci95_upper': ci_high
        })

    summary_df = pd.DataFrame(rows)

    description = (
        "For each culture (culture_id), this table reports the estimated effect of a one-standard-deviation "
        "increase in age (age_z) on the log-odds of choosing the majority option versus the reference "
        "(unchosen) option. The estimate for culture 1 is the main effect of age_z (reference culture). "
        "Estimates for culture c>1 are the sum of the main age effect and the age_x_culture_c interaction; "
        "standard errors, z-values, p-values and 95% CIs are computed using the covariance block for the "
        "majority outcome parameters. A statistically significant positive estimate indicates that older "
        "children in that culture are more likely to choose the majority option (compared to the reference), "
        "whereas a significant negative estimate indicates they are less likely."
    )

    return {"object": summary_df, "description": description}