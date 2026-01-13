def extract_final_answer(model_output):
    """
    Extracts genus comparison statistics from a fitted GLM (cluster-robust results allowed).
    Assumes the model used Treatment coding with "Homo sapiens" as the reference level,
    so coefficients for the non-human genera represent log-odds differences (genus vs Homo sapiens).

    Returns a dictionary:
      - "object": a dict with per-genus statistics (coef, se, p-value, 95% CI on log-odds,
                  odds-ratio and its 95% CI), plus summary booleans
      - "description": brief interpretation in plain language

    Example keys in returned object:
      {
        "per_genus": {
          "Pan": {"coef": ..., "se": ..., "p": ..., "ci": [low, high], "or": ..., "or_ci": [low, high]},
          "Pongo": {...},
          "Papio": {...}
        },
        "humans_higher_all": True/False,    # True if all non-human genera have significantly LOWER AMTL (coef<0 & p<0.05)
        "humans_higher_some": [list of genera where humans have significantly higher AMTL],
        "notes": "coefficients are log-odds for genus vs Homo sapiens (reference)"
      }
    """
    import numpy as np

    # Defensive: accept either a results wrapper or plain results
    res = model_output

    # Attempt to access parameter objects; raise informative error if missing
    try:
        params = res.params
        bse = res.bse
        pvalues = res.pvalues
        try:
            ci = res.conf_int()
        except Exception:
            # If conf_int method isn't available, compute approximate 95% CI from bse
            ci_lower = params - 1.96 * bse
            ci_upper = params + 1.96 * bse
            ci = np.column_stack([ci_lower, ci_upper])
    except Exception as e:
        raise ValueError("Provided model_output does not appear to be a fitted results object with params/bse/pvalues.") from e

    # Target non-human genera to extract (these were the levels in the original model)
    target_genera = ["Pan", "Pongo", "Papio"]

    per_genus = {}
    humans_higher_significant = []

    # params and pvalues may be a pandas Series with index names like
    # 'C(genus, Treatment(reference="Homo sapiens"))[T.Pan]' or similar.
    # We'll search for any parameter name containing the genus string.
    param_index = list(params.index) if hasattr(params, 'index') else None

    for g in target_genera:
        # Find parameter name containing the genus label
        matched_names = [name for name in (param_index or []) if g in str(name)]
        if not matched_names:
            # If no parameter found, skip but record None
            per_genus[g] = {
                "coef": None, "se": None, "p": None, "ci": [None, None],
                "or": None, "or_ci": [None, None],
                "note": f"No parameter matching genus '{g}' found in model parameters."
            }
            continue

        # If multiple matches (unlikely), take the first
        pname = matched_names[0]
        coef = float(params[pname])
        se = float(bse[pname]) if bse is not None else None
        pval = float(pvalues[pname]) if pvalues is not None else None

        # Confidence interval: res.conf_int() returns DataFrame or ndarray indexed same as params
        try:
            ci_vals = ci.loc[pname].tolist() if hasattr(ci, 'loc') else list(ci[param_index.index(pname)])
        except Exception:
            # Fallback: compute from coef +/- 1.96*se
            if se is not None:
                ci_vals = [coef - 1.96 * se, coef + 1.96 * se]
            else:
                ci_vals = [None, None]

        # Odds ratio and its CI
        try:
            or_val = float(np.exp(coef))
            or_ci = [float(np.exp(ci_vals[0])) if ci_vals[0] is not None else None,
                     float(np.exp(ci_vals[1])) if ci_vals[1] is not None else None]
        except Exception:
            or_val = None
            or_ci = [None, None]

        per_genus[g] = {
            "param_name": str(pname),
            "coef": coef,
            "se": se,
            "p": pval,
            "ci": ci_vals,
            "or": or_val,
            "or_ci": or_ci
        }

        # Interpretation: because Homo sapiens is the reference, a negative coef means that
        # the given non-human genus has LOWER log-odds (and thus lower probability) of AMTL
        # compared to humans. We flag genera where coef < 0 and p < 0.05.
        if (coef is not None) and (pval is not None) and (coef < 0) and (pval < 0.05):
            humans_higher_significant.append(g)

    humans_higher_all = (len(humans_higher_significant) == len([g for g in target_genera if per_genus[g].get("coef") is not None]))

    # Build description text
    if len(humans_higher_significant) == 0:
        conclusion = ("No clear evidence that modern humans (Homo sapiens) have higher AMTL than the "
                      "non-human genera included (Pan, Pongo, Papio): none of the genus coefficients "
                      "showed a statistically significant negative difference (p < 0.05).")
    elif humans_higher_all:
        conclusion = ("Yes — modern humans have higher AMTL than all three non-human genera (Pan, Pongo, Papio). "
                      "Each non-human genus shows a statistically significant lower AMTL compared to Homo sapiens "
                      "(coefficients < 0, p < 0.05).")
    else:
        conclusion = ("Partially: modern humans have significantly higher AMTL than some but not all non-human genera. "
                      f"Significant comparisons (humans higher): {humans_higher_significant}. "
                      "See per-genus statistics for details (coefficients are log-odds: negative means that genus has lower AMTL than humans).")

    result_object = {
        "per_genus": per_genus,
        "humans_higher_all": bool(humans_higher_all),
        "humans_higher_some": humans_higher_significant,
        "notes": ("Model coefficients are log-odds for (genus) minus (Homo sapiens). "
                  "Negative coefficient => genus has LOWER AMTL than Homo sapiens. "
                  "Odds ratios (OR = exp(coef)) and 95% CIs are also provided."),
    }

    return {"object": result_object, "description": conclusion}