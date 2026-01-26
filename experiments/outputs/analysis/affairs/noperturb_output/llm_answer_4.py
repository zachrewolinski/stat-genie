def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, confidence intervals, and effect sizes
    for the effect of 'children_yes' (and its interaction with gender if present) from the
    provided model_output dict. Also returns group descriptives by children.

    Returns a dict with keys:
      - "object": dict with per-model numeric summaries and an overall conclusion decision
      - "description": brief plain-language interpretation of those results
    """
    import numpy as np

    def safe_get_param_stats(res, target_terms):
        """
        Given a statsmodels results object `res` and a list of strings `target_terms`
        (e.g., ['children_yes'] or ['children_yes','gender_male'] for an interaction),
        attempt to find matching parameter names and return a dict of stats.
        Matching tries exact name first, then parameter names that contain all terms.
        Returns None if no matching parameter is found.
        """
        if res is None:
            return None
        try:
            params = res.params
            bse = res.bse
            pvalues = res.pvalues
            conf = res.conf_int()
        except Exception:
            # Not a statsmodels-like results object
            return None

        # list of parameter names (index)
        names = list(params.index)

        # 1) try exact matches
        joined = ':'.join(target_terms)
        if joined in names:
            name = joined
        elif target_terms[0] in names:
            # prefer exact single-term match if target_terms length==1
            if len(target_terms) == 1:
                name = target_terms[0]
            else:
                # look for any name that contains all terms
                name = None
                for n in names:
                    if all(t in n for t in target_terms):
                        name = n
                        break
        else:
            # 2) fallback: find first parameter name that contains all target terms
            name = None
            for n in names:
                if all(t in n for t in target_terms):
                    name = n
                    break

        if name is None:
            return None

        try:
            coef = float(params[name])
            se = float(bse[name]) if name in bse.index else float(bse.loc[name])
            p = float(pvalues[name]) if name in pvalues.index else float(pvalues.loc[name])
            ci_low, ci_high = float(conf.loc[name, 0]), float(conf.loc[name, 1])
        except Exception:
            # If any indexing fails, return None
            return None

        return {"param_name": name, "coef": coef, "se": se, "pvalue": p, "ci_lower": ci_low, "ci_upper": ci_high}

    summary = {}
    models = ['ols', 'neg_bin', 'zinb']
    for m in models:
        res = model_output.get(m)
        if res is None:
            summary[m] = {"present": False, "note": "model not present or failed"}
            continue

        # Extract main 'children_yes' effect
        main = safe_get_param_stats(res, ['children_yes'])

        # Extract interaction 'children_yes:gender_male' or similar if present
        interaction = safe_get_param_stats(res, ['children_yes', 'gender_male'])

        # For count models, compute exp(coef) and exp(CI) as IRR if coef available
        irr = None
        irr_ci = None
        if main is not None:
            try:
                irr = float(np.exp(main['coef']))
                irr_ci = (float(np.exp(main['ci_lower'])), float(np.exp(main['ci_upper'])))
            except Exception:
                irr = None
                irr_ci = None

        summary[m] = {
            "present": True,
            "main": main,
            "interaction": interaction,
            "irr": irr,
            "irr_ci": irr_ci
        }

    # Add descriptive stats if present
    desc = model_output.get('descriptive_by_children')
    if desc is not None:
        summary['descriptive_by_children'] = desc

    # Formulate simple verdicts per model about whether 'having children' decreases affairs
    verdicts = {}
    for m in models:
        info = summary.get(m)
        if not info or not info.get("present"):
            verdicts[m] = {"conclusion": "no_model", "reason": "model not available"}
            continue
        main = info.get("main")
        if main is None:
            verdicts[m] = {"conclusion": "no_estimate", "reason": "no 'children_yes' parameter found in model"}
            continue

        coef = main['coef']
        p = main['pvalue']
        # significance at alpha=0.05
        sig = (p < 0.05)
        if coef < 0:
            if sig:
                conclusion = "decrease_significant"
                reason = f"Coefficient {coef:.4f} (p={p:.3g}) indicates a statistically significant decrease."
            else:
                conclusion = "decrease_nonsig"
                reason = f"Coefficient {coef:.4f} (p={p:.3g}) indicates a non-significant decrease."
        elif coef > 0:
            if sig:
                conclusion = "increase_significant"
                reason = f"Coefficient {coef:.4f} (p={p:.3g}) indicates a statistically significant increase."
            else:
                conclusion = "increase_nonsig"
                reason = f"Coefficient {coef:.4f} (p={p:.3g}) indicates a non-significant increase."
        else:
            conclusion = "no_effect"
            reason = f"Coefficient is {coef:.4f} (p={p:.3g})."

        # For count models, add IRR interpretation if available
        irr = info.get("irr")
        if irr is not None:
            reason += f" IRR = {irr:.3f}"
            if info.get("irr_ci") is not None:
                reason += f" (CI {info['irr_ci'][0]:.3f} - {info['irr_ci'][1]:.3f})"

        verdicts[m] = {"conclusion": conclusion, "reason": reason, "pvalue": p, "coef": coef}

    # Aggregate a simple overall conclusion:
    # Count models that show significant decrease vs significant increase
    sig_decrease = sum(1 for v in verdicts.values() if v.get('conclusion') == 'decrease_significant')
    sig_increase = sum(1 for v in verdicts.values() if v.get('conclusion') == 'increase_significant')

    if sig_decrease > 0 and sig_increase == 0:
        overall = "Having children appears to decrease engagement in extramarital affairs (statistically significant in one or more models)."
    elif sig_increase > 0 and sig_decrease == 0:
        overall = "Having children does NOT decrease engagement in extramarital affairs; models show an increase (statistically significant in one or more models)."
    elif sig_decrease == 0 and sig_increase == 0:
        overall = "There is no consistent statistically significant evidence that having children decreases engagement in extramarital affairs across the fitted models."
    else:
        overall = "Mixed evidence: some models show a significant decrease and others a significant increase."

    # Compose the returned object
    results_object = {
        "per_model_summary": summary,
        "model_verdicts": verdicts,
        "overall_conclusion": overall
    }

    # Short description
    description_lines = [
        "Extracted per-model estimates for the 'children_yes' effect (coefficient, SE, p-value, 95% CI).",
        "For count models (Negative Binomial / ZINB) the exponentiated coefficient (IRR) is also provided where available.",
        "A simple per-model verdict (significant increase/decrease/no effect) and an overall conclusion are included."
    ]
    description = " ".join(description_lines) + " Overall conclusion: " + overall

    return {"object": results_object, "description": description}