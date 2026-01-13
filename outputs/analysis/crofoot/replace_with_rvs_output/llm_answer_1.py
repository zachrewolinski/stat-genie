def extract_final_answer(model_output):
    """
    Extracts coefficients, p-values, confidence intervals, and computes the
    marginal effect of `size_diff` (log-odds scale) at each ContestLocation
    level from a fitted statsmodels logistic regression (possibly cluster-robust).

    Returns:
      {
        "object": {
            "params": {name: value, ...},
            "pvalues": {name: value, ...},
            "conf_int": {name: [low, high], ...},
            "marginal_effects_by_location": {
                location_name: {
                    "effect_log_odds": float,
                    "se": float or None,
                    "z": float or None,
                    "p": float or None,
                    "95%_ci": [low, high] or [None, None]
                }, ...
            },
            "reference_level": str or None
        },
        "description": "Plain-English interpretation of the extracted results."
      }
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import norm

    res = model_output

    # Validate we have parameter estimates
    if not hasattr(res, "params"):
        raise ValueError("model_output has no .params attribute; expected a statsmodels results object.")

    params = res.params.copy()
    pvalues = res.pvalues.copy() if hasattr(res, "pvalues") else pd.Series(index=params.index, data=[None]*len(params))
    # conf_int may require alpha; most results provide conf_int() method
    try:
        ci_df = res.conf_int()  # returns DataFrame with two columns
        ci_df.columns = ["2.5%", "97.5%"]
    except Exception:
        # fallback: put Nones
        ci_df = pd.DataFrame(index=params.index, columns=["2.5%", "97.5%"])
        ci_df.loc[:, :] = None

    # Attempt to get covariance matrix for variance of linear combinations
    cov = None
    try:
        cov = res.cov_params()
    except Exception:
        cov = None

    # Identify variable names of interest
    param_names = list(params.index)

    # Standard name for focal predictor
    size_name = None
    for n in param_names:
        if n == "size_diff":
            size_name = n
            break
    if size_name is None:
        # try variations
        for n in param_names:
            if n.endswith("size_diff") or "size_diff" in n:
                # prefer exact match but otherwise choose the one that is exactly 'size_diff'
                size_name = n
                break

    if size_name is None:
        raise ValueError("Could not find a parameter corresponding to 'size_diff' in model parameters. Found: " + ", ".join(param_names))

    # Find contest location main-effect dummy names and interaction names
    contest_main = {}
    contest_inter = {}
    # Patterns to look for: 'ContestLocation[T.SomeLevel]' for main; interaction may be 'size_diff:ContestLocation[T.SomeLevel]' or vice versa
    for n in param_names:
        if "ContestLocation" in n and ":" not in n:
            # main effect for a level (dummy)
            contest_main[n] = params[n]
        if "ContestLocation" in n and ":" in n and "size_diff" in n:
            # interaction term
            contest_inter[n] = params[n]

    # Try to obtain full list of contest location levels from the original dataframe, if available
    reference_level = None
    levels_from_data = None
    try:
        df = res.model.data.frame
        if "ContestLocation" in df.columns:
            ser = pd.Categorical(df["ContestLocation"])
            levels_from_data = list(ser.categories)
            # Determine which levels appear as dummies in params: these will be 'ContestLocation[T.level]'
            dummy_levels = []
            for lvl in levels_from_data:
                key = f"ContestLocation[T.{lvl}]"
                if key in param_names:
                    dummy_levels.append(lvl)
            # reference level is the one not present as a dummy (if exactly one)
            ref = [lvl for lvl in levels_from_data if lvl not in dummy_levels]
            if len(ref) == 1:
                reference_level = ref[0]
            else:
                # ambiguous or ordering unknown
                reference_level = None
    except Exception:
        levels_from_data = None
        reference_level = None

    # If we couldn't get levels from data, infer levels from parameter names
    inferred_levels = []
    for n in param_names:
        if n.startswith("ContestLocation[T."):
            # extract between 'ContestLocation[T.' and ']'
            try:
                lvl = n.split("ContestLocation[T.")[1].split("]")[0]
                inferred_levels.append(lvl)
            except Exception:
                continue
    inferred_levels = list(dict.fromkeys(inferred_levels))  # unique preserve order

    # Build list of all levels we can report: those inferred plus the reference (if known)
    levels_report = []
    if levels_from_data is not None:
        levels_report = list(levels_from_data)
    elif inferred_levels:
        # we only know the dummy levels; we denote a reference as omitted
        levels_report = list(inferred_levels)
        # add a placeholder for the omitted/reference
        levels_report.append("(reference omitted)")
    else:
        # no contest location info found
        levels_report = ["(no ContestLocation info in model)"]

    # Helper: find interaction param name for a given level (any order)
    def find_interaction_name_for_level(level):
        candidates = []
        target_fragment = f"ContestLocation[T.{level}]"
        for n in param_names:
            if target_fragment in n and "size_diff" in n and ":" in n:
                candidates.append(n)
        if len(candidates) >= 1:
            return candidates[0]
        return None

    # Prepare results per location: marginal effect of size_diff on log-odds scale
    marg_effects = {}
    coef_size = float(params[size_name])
    var_size = None
    if cov is not None and size_name in cov.index:
        var_size = float(cov.loc[size_name, size_name])

    for lvl in levels_report:
        if lvl == "(no ContestLocation info in model)":
            marg_effects[lvl] = {
                "effect_log_odds": coef_size,
                "se": float(np.sqrt(var_size)) if var_size is not None else None,
                "z": (coef_size / np.sqrt(var_size)) if var_size is not None and var_size > 0 else None,
                "p": (2 * (1 - norm.cdf(abs(coef_size / np.sqrt(var_size))))) if var_size is not None and var_size > 0 else None,
                "95%_ci": [float(ci_df.loc[size_name, "2.5%"]) if size_name in ci_df.index else None,
                           float(ci_df.loc[size_name, "97.5%"]) if size_name in ci_df.index else None]
            }
            continue

        if lvl == "(reference omitted)":
            # we can't identify which actual level is reference, but we can report the base effect
            effect = coef_size
            se = float(np.sqrt(var_size)) if var_size is not None else None
            z = (effect / se) if se is not None and se > 0 else None
            p = (2 * (1 - norm.cdf(abs(z)))) if z is not None else None
            ci_low = float(ci_df.loc[size_name, "2.5%"]) if size_name in ci_df.index else None
            ci_high = float(ci_df.loc[size_name, "97.5%"]) if size_name in ci_df.index else None
            marg_effects[lvl] = {
                "effect_log_odds": float(effect),
                "se": se,
                "z": float(z) if z is not None else None,
                "p": float(p) if p is not None else None,
                "95%_ci": [ci_low, ci_high]
            }
            continue

        # For a concrete named level:
        inter_name = find_interaction_name_for_level(lvl)
        if inter_name is None:
            # No interaction for this level -> effect is the main size_diff coefficient
            effect = coef_size
            # use var_size
            se = float(np.sqrt(var_size)) if var_size is not None else None
            z = (effect / se) if se is not None and se > 0 else None
            p = (2 * (1 - norm.cdf(abs(z)))) if z is not None else None
            ci_low = float(ci_df.loc[size_name, "2.5%"]) if size_name in ci_df.index else None
            ci_high = float(ci_df.loc[size_name, "97.5%"]) if size_name in ci_df.index else None
            marg_effects[lvl] = {
                "effect_log_odds": float(effect),
                "se": se,
                "z": float(z) if z is not None else None,
                "p": float(p) if p is not None else None,
                "95%_ci": [ci_low, ci_high]
            }
        else:
            # effect = coef(size_diff) + coef(interaction)
            coef_inter = float(params[inter_name])
            effect = coef_size + coef_inter
            # variance of sum = var(size) + var(inter) + 2*cov(size,inter)
            se = None
            z = None
            p = None
            ci_low = None
            ci_high = None
            if cov is not None and inter_name in cov.index and size_name in cov.index:
                v_size = float(cov.loc[size_name, size_name])
                v_inter = float(cov.loc[inter_name, inter_name])
                covar = float(cov.loc[size_name, inter_name])
                var_sum = v_size + v_inter + 2.0 * covar
                if var_sum >= 0:
                    se = float(np.sqrt(var_sum))
                    if se > 0:
                        z = effect / se
                        p = 2 * (1 - norm.cdf(abs(z)))
                        # approximate CI on log-odds
                        ci_low = effect - norm.ppf(0.975) * se
                        ci_high = effect + norm.ppf(0.975) * se
            # fallback: if cov missing, try to approximate with naive combination of CIs (not ideal)
            if se is None:
                marg_effects[lvl] = {
                    "effect_log_odds": float(effect),
                    "se": None,
                    "z": None,
                    "p": None,
                    "95%_ci": [None, None]
                }
            else:
                marg_effects[lvl] = {
                    "effect_log_odds": float(effect),
                    "se": se,
                    "z": float(z),
                    "p": float(p),
                    "95%_ci": [float(ci_low), float(ci_high)]
                }

    # Build dictionaries for params, pvalues, ci
    params_dict = {k: float(v) for k, v in params.items()}
    pvalues_dict = {k: float(v) for k, v in pvalues.items()}
    ci_dict = {k: [float(ci_df.loc[k, "2.5%"]) if k in ci_df.index and ci_df.loc[k, "2.5%"] is not None else None,
                   float(ci_df.loc[k, "97.5%"]) if k in ci_df.index and ci_df.loc[k, "97.5%"] is not None else None]
               for k in params.index}
    ci_dict = dict(zip(params.index.tolist(), ci_dict))

    # Compose a human-readable description
    # We'll report the main coefficient for size_diff and whether any interaction modified it
    desc_lines = []
    desc_lines.append("Extracted model estimates related to relative group size (size_diff) and its interaction with ContestLocation.")
    desc_lines.append(f"Main coefficient for 'size_diff' (log-odds change per unit size_diff): {params_dict.get(size_name):.4f}")
    p_main = pvalues_dict.get(size_name, None)
    if p_main is not None:
        desc_lines.append(f"  - p-value for main size_diff term: {p_main:.4g}")
    # Report interactions found
    if len(contest_inter) > 0:
        desc_lines.append("Interaction terms detected between size_diff and ContestLocation levels. Marginal effects (log-odds) of size_diff by location are provided under 'object' -> 'marginal_effects_by_location'.")
    else:
        desc_lines.append("No interaction terms between size_diff and ContestLocation were detected in the fitted model. Effect of size_diff is the same across locations as modeled.")
    desc_lines.append("Interpretation: effects are on the log-odds scale. Positive effect_log_odds means that when the focal group is relatively larger than the opponent, the log-odds of the focal group winning increase. To convert to odds ratio, exponentiate the effect (exp(effect_log_odds)). For approximate change in probability, compute predicted probabilities at relevant baseline probabilities or use marginal effects on probability scale (not computed here).")

    description = " ".join(desc_lines)

    out = {
        "object": {
            "params": params_dict,
            "pvalues": pvalues_dict,
            "conf_int": ci_dict,
            "marginal_effects_by_location": marg_effects,
            "reference_level": reference_level
        },
        "description": description
    }
    return out