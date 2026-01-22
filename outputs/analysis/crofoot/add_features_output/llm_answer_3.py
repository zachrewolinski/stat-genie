def extract_final_answer(model_output):
    """
    Extracts coefficients, robust SEs, z-stats, p-values, and 95% CIs for:
      - the main effect of RelSize_z
      - the interaction terms RelSize_z:C(ContestLocation)[T.<level>]
      - the implied marginal effect of RelSize_z at each ContestLocation level
    Returns:
      {
        "object": {
           "param_table": pd.DataFrame,         # rows for main + interactions (coef, se, z, p, ci_lower, ci_upper)
           "marginal_effects": pd.DataFrame,    # rows for each location level (coef, se, z, p, ci_lower, ci_upper)
        },
        "description": str  # brief interpretation
      }
    """
    import re
    import numpy as np
    import pandas as pd
    from scipy import stats

    # Basic parameter objects
    params = model_output.params
    param_names = [str(n) for n in params.index]
    # Covariance matrix (robust cov if model_output already contains robust results)
    cov = model_output.cov_params()
    # Ensure cov is an ndarray ordered to param_names
    if isinstance(cov, pd.DataFrame):
        cov_mat = cov.reindex(index=param_names, columns=param_names).values
    else:
        cov_mat = np.asarray(cov)

    # Helper to get index of a parameter name (exact match)
    name_to_idx = {name: i for i, name in enumerate(param_names)}

    # Find main RelSize_z parameter
    if 'RelSize_z' in name_to_idx:
        main_name = 'RelSize_z'
    else:
        # Try to find any parameter that equals 'RelSize_z' or starts with it (defensive)
        candidates = [n for n in param_names if n.split(':')[0] == 'RelSize_z' and ':' not in n]
        if len(candidates) > 0:
            main_name = candidates[0]
        else:
            raise ValueError("Could not find main effect parameter 'RelSize_z' in model parameters.")

    main_idx = name_to_idx[main_name]

    # Find interaction parameters that involve RelSize_z and ContestLocation
    interaction_pattern = re.compile(r"RelSize_z:.*ContestLocation|ContestLocation.*:RelSize_z")
    interaction_names = [n for n in param_names if interaction_pattern.search(n)]

    # Extract level labels for interactions (if possible)
    interaction_levels = []
    for n in interaction_names:
        m = re.search(r"C\(ContestLocation\)\[T\.([^]]+)\]", n)
        if not m:
            # try alternate pattern without C(...)
            m = re.search(r"ContestLocation\[T\.([^]]+)\]", n)
        interaction_levels.append(m.group(1) if m else n)

    # Build parameter table for main + interactions
    rows = []
    for n in [main_name] + interaction_names:
        idx = name_to_idx[n]
        coef = float(params.iloc[idx])
        # SE from covariance diagonal
        se = float(np.sqrt(cov_mat[idx, idx]))
        z = coef / se if se != 0 else np.nan
        p = 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
        ci_low = coef - 1.96 * se
        ci_upp = coef + 1.96 * se
        rows.append({
            "param": n,
            "coef": coef,
            "se": se,
            "z": z,
            "p": p,
            "ci_lower": ci_low,
            "ci_upper": ci_upp
        })
    param_table = pd.DataFrame(rows).set_index('param')

    # Determine all observed ContestLocation levels from the model data if available
    baseline_label = None
    observed_levels = None
    try:
        df = model_output.model.data.frame
        if 'ContestLocation' in df.columns:
            observed_levels = list(pd.unique(df['ContestLocation']))
            # baseline is the one without an explicit indicator param (i.e., omitted category)
            baseline_candidates = []
            for lvl in observed_levels:
                indicator_name = f"C(ContestLocation)[T.{lvl}]"
                if indicator_name not in param_names:
                    baseline_candidates.append(lvl)
            if len(baseline_candidates) == 1:
                baseline_label = baseline_candidates[0]
    except Exception:
        # If frame not available, fall back to inferring baseline from parameter names:
        pass

    # If baseline not found, attempt to infer from parameter names:
    if baseline_label is None and observed_levels is None:
        # Try to extract all levels from param names and set baseline as the one not present
        # Find all levels mentioned in C(ContestLocation)[T.<level>] tokens
        mentioned = []
        for n in param_names:
            m = re.search(r"C\(ContestLocation\)\[T\.([^]]+)\]", n)
            if m:
                mentioned.append(m.group(1))
        # We cannot know full set reliably; treat baseline generically as "reference (omitted) level"
        baseline_label = "reference (omitted)"
    elif baseline_label is None and observed_levels is not None:
        # Determine which observed level is omitted
        mentioned = []
        for n in param_names:
            m = re.search(r"C\(ContestLocation\)\[T\.([^]]+)\]", n)
            if m:
                mentioned.append(m.group(1))
        omitted = [lvl for lvl in observed_levels if lvl not in mentioned]
        baseline_label = omitted[0] if len(omitted) == 1 else "reference (omitted)"

    # Compute marginal effect of RelSize_z at each location level:
    # - For baseline: effect = main coef
    # - For each other level L: effect = main coef + interaction coef for L (if present)
    marg_rows = []
    # baseline first
    a = np.zeros(len(param_names))
    a[main_idx] = 1.0
    coef_baseline = float(np.dot(a, params.values))
    var_baseline = float(a @ cov_mat @ a)
    se_baseline = float(np.sqrt(var_baseline))
    z_baseline = coef_baseline / se_baseline if se_baseline != 0 else np.nan
    p_baseline = 2 * (1 - stats.norm.cdf(abs(z_baseline))) if not np.isnan(z_baseline) else np.nan
    marg_rows.append({
        "location": str(baseline_label),
        "coef": coef_baseline,
        "se": se_baseline,
        "z": z_baseline,
        "p": p_baseline,
        "ci_lower": coef_baseline - 1.96 * se_baseline,
        "ci_upper": coef_baseline + 1.96 * se_baseline
    })

    # Now each interaction level
    for name, lvl in zip(interaction_names, interaction_levels):
        a = np.zeros(len(param_names))
        a[main_idx] = 1.0
        idx_int = name_to_idx[name]
        a[idx_int] = 1.0
        coef_val = float(np.dot(a, params.values))
        var_val = float(a @ cov_mat @ a)
        se_val = float(np.sqrt(var_val))
        z_val = coef_val / se_val if se_val != 0 else np.nan
        p_val = 2 * (1 - stats.norm.cdf(abs(z_val))) if not np.isnan(z_val) else np.nan
        marg_rows.append({
            "location": str(lvl),
            "coef": coef_val,
            "se": se_val,
            "z": z_val,
            "p": p_val,
            "ci_lower": coef_val - 1.96 * se_val,
            "ci_upper": coef_val + 1.96 * se_val
        })

    marginal_effects = pd.DataFrame(marg_rows).set_index('location')

    description_lines = [
        "Extracted results concern the effect of relative group size (RelSize_z) on the log-odds",
        "that the focal group wins, and how that effect differs by ContestLocation.",
        "- 'param_table' shows the estimated coefficient, robust SE, z, p-value, and 95% CI for",
        "  the main RelSize_z term and its interactions with ContestLocation.",
        "- 'marginal_effects' gives the implied effect of a one-unit increase in RelSize_z (here standardized)",
        "  on the log-odds of focal victory at each contest location. Positive coef => higher log-odds of winning",
        "  with increasing relative group size. P-values indicate whether effects differ from zero.",
        "",
        "Interpretation guidance (example):",
        "- If the marginal effect for 'FocalHome' (or the baseline/omitted level) is positive and statistically",
        "  significant (p < 0.05), then larger relative group size increases the focal group's probability of winning",
        "  when contests occur nearer the focal group's center.",
        "- If the interaction term for 'OtherHome' is negative and statistically significant, the positive effect",
        "  of RelSize_z is reduced (or reversed) when contests occur nearer the other group's center."
    ]
    description = "\n".join(description_lines)

    return {
        "object": {
            "param_table": param_table,
            "marginal_effects": marginal_effects
        },
        "description": description
    }