def extract_final_answer(model_output):
    """
    Extract statistics from a fitted statsmodels GLMResults-like object (possibly
    with clustered robust covariance) to answer whether modern humans (Homo)
    have higher AMTL than Pan, Pongo, and Papio, after controlling for covariates.
    
    Returns:
      {
        "object": {
            "decision": True/False/None,
            "reason": short reason string,
            "predicted_probabilities": { genus_label: { "lp":..., "prob":..., "prob_95ci": (low,high) }, ... },
            "pairwise_comparisons": {
                "Homo_vs_Pan": {
                    "diff_logit": ...,
                    "se_diff": ...,
                    "z": ...,
                    "p_value": ...,
                    "Homo_prob": ...,
                    "Other_prob": ...,
                    "interpretation": ...
                }, ...
            }
        },
        "description": "Brief explanation..."
      }
    Notes:
      - This function assumes the model used column names like 'const' (or 'Intercept')
        and genus dummy names with prefix 'Genus_' (e.g., 'Genus_Pan', 'Genus_Homo sapiens').
      - If the exact label used for Homo in the data cannot be identified unambiguously,
        the function will return the computed probabilities for all observed genus levels
        and set decision to None.
    """
    import numpy as np
    from math import sqrt
    from scipy.special import expit
    from scipy.stats import norm

    res = model_output

    # Extract params and covariance matrix
    try:
        params = res.params.copy()
        cov = res.cov_params().copy()
    except Exception as e:
        raise ValueError(f"Cannot extract params/covariance from model_output: {e}")

    param_names = list(params.index)

    # Find intercept name
    intercept_candidates = ['const', 'Const', 'INTERCEPT', 'Intercept', 'intercept']
    intercept_name = None
    for cand in intercept_candidates:
        if cand in param_names:
            intercept_name = cand
            break
    if intercept_name is None:
        # fallback: use first parameter as intercept if its name contains no '=' and is not a predictor pattern
        # but we try a safer approach: if 'const' not found, raise a helpful error
        raise ValueError("Intercept ('const' or 'Intercept') not found among model parameters. "
                         f"Found parameters: {param_names}")

    # Identify genus dummy columns
    genus_dummy_names = [n for n in param_names if n.startswith('Genus_')]
    genus_levels = [n.split('Genus_', 1)[1] for n in genus_dummy_names]

    # Build weight vector helper to compute linear predictor and its variance for any genus label
    def make_weight_for_genus(dummy_name):
        """
        Returns weight vector (length = len(params)) that when dotted with params yields
        the linear predictor for a specimen with:
          - intercept = 1
          - genus = dummy_name (if dummy_name is None => reference genus with no dummy)
          - all continuous covariates set to their means (they were centered => 0)
          - all toothclass dummies = reference (all zero)
        """
        w = np.zeros(len(param_names), dtype=float)
        # intercept weight
        w[param_names.index(intercept_name)] = 1.0
        # genus effect if present
        if dummy_name is not None:
            if dummy_name in param_names:
                w[param_names.index(dummy_name)] = 1.0
            else:
                # dummy not in params -> treat as reference (weight remains zero)
                pass
        return w

    # Compute lp, se, prob, and 95% CI for a given weight vector
    def compute_stats_for_weight(w):
        vals = params.values
        lp = float(w.dot(vals))
        var_lp = float(w @ cov.values @ w)
        se_lp = sqrt(max(var_lp, 0.0))
        lp_low = lp - 1.96 * se_lp
        lp_high = lp + 1.96 * se_lp
        prob = float(expit(lp))
        prob_low = float(expit(lp_low))
        prob_high = float(expit(lp_high))
        return {"lp": lp, "se_lp": se_lp, "prob": prob, "prob_95ci": (prob_low, prob_high)}

    # Prepare mapping of observed genus labels -> dummy column names and stats
    observed_genus_map = {}  # display_label -> dummy_column_name or None for reference
    for lvl, dummy in zip(genus_levels, genus_dummy_names):
        observed_genus_map[lvl] = dummy

    # Try to detect the dropped/reference genus by comparing to expected genera
    expected_core = ['Homo', 'Pan', 'Pongo', 'Papio']
    # For matching we do case-insensitive substring checks
    present_expected = {g: any(g.lower() in lvl.lower() for lvl in observed_genus_map.keys()) for g in expected_core}
    # If a genus from expected_core is not present among the dummies, it is likely the reference (dropped)
    missing_expected = [g for g, present in present_expected.items() if not present]

    reference_label = None
    if len(missing_expected) == 1:
        # identify the likely reference genus
        reference_label = missing_expected[0]
    else:
        # If we cannot uniquely identify reference among expected set, we will set reference_label to None
        reference_label = None

    # If we determined a reference label and it's not in observed_genus_map, include it mapped to None
    if reference_label is not None and reference_label not in observed_genus_map:
        observed_genus_map[reference_label] = None

    # Compute stats for each observed genus label (including the reference one if identified)
    predicted = {}
    for label, dummy in observed_genus_map.items():
        w = make_weight_for_genus(dummy)
        predicted[label] = compute_stats_for_weight(w)

    # If we didn't detect reference_label but there are exactly 3 dummy columns, try infer the dropped label
    if reference_label is None and len(genus_levels) == 3:
        # try to infer by checking which of expected_core is not matched (even if multiple substrings)
        unmatched = []
        for g in expected_core:
            if not any(g.lower() in lvl.lower() for lvl in observed_genus_map.keys()):
                unmatched.append(g)
        if len(unmatched) == 1:
            reference_label = unmatched[0]
            if reference_label not in observed_genus_map:
                observed_genus_map[reference_label] = None
                predicted[reference_label] = compute_stats_for_weight(make_weight_for_genus(None))

    # Now attempt pairwise comparisons between Homo and each non-human genus (Pan, Pongo, Papio)
    comparisons = {}
    decision = None
    reason = ""
    # Only proceed if we can identify which label corresponds to Homo
    homo_label = None
    # find an observed key that looks like Homo if present
    for label in list(observed_genus_map.keys()):
        if 'homo' in label.lower() or 'human' in label.lower():
            homo_label = label
            break
    # If Homo label not found among keys but reference_label is 'Homo', set homo_label to that
    if homo_label is None and reference_label is not None and reference_label.lower().startswith('homo'):
        homo_label = reference_label

    # We'll collect results for the three target non-human genera when possible
    target_nonhuman = ['Pan', 'Pongo', 'Papio']
    all_successful = True
    homo_vs_results = {}
    if homo_label is None:
        decision = None
        reason = ("Could not unambiguously identify which model genus label corresponds to modern humans "
                  "(Homo). Returning predicted probabilities for observed genus levels instead.")
    else:
        # compute weight vectors for Homo and each target genus
        # We need the full params vector and cov matrix
        param_vals = params.values
        cov_vals = cov.values
        # helper to build weight vector for given label name (which maps to dummy or None)
        def w_for_label(label):
            if label not in observed_genus_map:
                return None
            return make_weight_for_genus(observed_genus_map[label])

        w_homo = w_for_label(homo_label)
        stats_homo = compute_stats_for_weight(w_homo)

        # Iterate comparisons
        comparisons_summary = {}
        for other in target_nonhuman:
            if other not in observed_genus_map:
                # The other genus label not observed -> cannot compute
                all_successful = False
                comparisons_summary[other] = {
                    "available": False,
                    "reason": f"Genus '{other}' not present among model genus dummies/labels; cannot compare."
                }
                continue
            w_other = w_for_label(other)
            # difference weight vector
            w_diff = w_homo - w_other
            diff = float(w_diff.dot(param_vals))
            var_diff = float(w_diff @ cov_vals @ w_diff)
            se_diff = sqrt(max(var_diff, 0.0))
            if se_diff == 0:
                z = np.nan
                pval = np.nan
            else:
                z = diff / se_diff
                pval = float(2.0 * (1.0 - norm.cdf(abs(z))))
            # compute probs
            stats_other = compute_stats_for_weight(w_other)
            # Interpretation: positive diff means Homo log-odds > other (higher AMTL in Homo)
            interpretation = ""
            if np.isnan(pval):
                interpretation = "Comparison not available (zero variance)."
            else:
                if pval < 0.05:
                    if diff > 0:
                        interpretation = "Homo has significantly higher AMTL than " + other
                    else:
                        interpretation = other + " has significantly higher AMTL than Homo"
                else:
                    interpretation = "No statistically significant difference between Homo and " + other
            comparisons_summary[other] = {
                "available": True,
                "diff_logit": diff,
                "se_diff": se_diff,
                "z": z,
                "p_value": pval,
                "Homo_prob": stats_homo["prob"],
                "Other_prob": stats_other["prob"],
                "Homo_prob_95ci": stats_homo["prob_95ci"],
                "Other_prob_95ci": stats_other["prob_95ci"],
                "interpretation": interpretation
            }

        # Decide final answer: require that for all three non-human genera comparisons are available
        # and that Homo has higher AMTL (diff>0) AND p<0.05 for each.
        if all(comparisons_summary.get(g, {}).get("available", False) for g in target_nonhuman):
            homo_higher_all = all((comparisons_summary[g]["diff_logit"] > 0 and comparisons_summary[g]["p_value"] < 0.05)
                                  for g in target_nonhuman)
            if homo_higher_all:
                decision = True
                reason = "Homo has higher AMTL than Pan, Pongo, and Papio (all differences positive and p<0.05)."
            else:
                decision = False
                reason = "Homo does not have consistently higher AMTL than all three non-human genera (one or more comparisons non-significant or reversed)."
        else:
            decision = None
            reason = "Could not compute all three pairwise comparisons between Homo and Pan/Pongo/Papio (some genera labels missing in model dummies)."

        comparisons = comparisons_summary

    # Prepare output object
    out_object = {
        "decision": decision,
        "reason": reason,
        "predicted_probabilities": predicted,
        "pairwise_comparisons": comparisons
    }

    desc_lines = [
        "Extracted predicted AMTL probabilities (with 95% CI) for each genus-level linear predictor (covariates at mean/reference).",
        "Also performed pairwise logit-scale comparisons (Homo vs each of Pan, Pongo, Papio) when genus labels could be matched.",
        "Decision is True if Homo shows higher AMTL than each non-human genus with p<0.05; False if not; None if comparison not possible/ambiguous."
    ]
    description = " ".join(desc_lines)

    return {"object": out_object, "description": description}