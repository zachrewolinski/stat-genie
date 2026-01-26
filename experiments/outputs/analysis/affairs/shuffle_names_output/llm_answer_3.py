def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of HasChildren on affairs from the model_output dict
    returned by the modeling function.

    Returns a dict with keys:
      - "object": a dict with extracted numeric results (if available) for logit and poisson models,
                  plus a short 'conclusion' statement when possible.
      - "description": a human-readable explanation of what was extracted and its interpretation
                       for the research question.

    The function handles cases where models failed to fit (None) and returns an appropriate
    explanatory message in that case.
    """
    import math

    result = {
        "logit": None,
        "poisson": None,
        "conclusion": None
    }

    notes = []

    # Helper to build stats dict and compute OR/IRR and 95% CI if possible
    def build_stats(coef, se, pval, model_type):
        stats = {"coef": coef, "se": se, "pvalue": pval}
        try:
            if coef is not None:
                # Compute exponentiated effect: odds ratio for logit, incidence rate ratio for Poisson
                stats["exp_coef"] = math.exp(coef)
            if coef is not None and se is not None:
                z = 1.96
                lower = coef - z * se
                upper = coef + z * se
                stats["ci_95"] = (lower, upper)
                stats["exp_ci_95"] = (math.exp(lower), math.exp(upper))
        except Exception:
            # If something fails (e.g., non-numeric), ignore extra computations
            pass
        stats["interpretation"] = (
            "log-odds (coef) for HasChildren; exp_coef is the odds ratio"
            if model_type == "logit" else
            "log-rate (coef) for HasChildren; exp_coef is the incidence rate ratio"
        )
        return stats

    # 1) Try to extract from the convenient 'summary' entry if present
    summary = model_output.get("summary") if isinstance(model_output, dict) else None
    if summary:
        logit_sum = summary.get("logit_HasChildren")
        pois_sum = summary.get("poisson_HasChildren")
        if logit_sum:
            # Expect keys 'coef', 'se', 'pvalue'
            result["logit"] = build_stats(
                coef=logit_sum.get("coef"),
                se=logit_sum.get("se"),
                pval=logit_sum.get("pvalue"),
                model_type="logit"
            )
        if pois_sum:
            result["poisson"] = build_stats(
                coef=pois_sum.get("coef"),
                se=pois_sum.get("se"),
                pval=pois_sum.get("pvalue"),
                model_type="poisson"
            )

    # 2) If summary missing details, try to extract directly from model objects (if any)
    # logit_result
    if result["logit"] is None and model_output.get("logit_result") is not None:
        try:
            res = model_output["logit_result"]
            coef = float(res.params["HasChildren"]) if "HasChildren" in res.params.index else None
            se = float(res.bse["HasChildren"]) if hasattr(res, "bse") and "HasChildren" in res.bse.index else None
            pval = float(res.pvalues["HasChildren"]) if hasattr(res, "pvalues") and "HasChildren" in res.pvalues.index else None
            result["logit"] = build_stats(coef, se, pval, "logit")
        except Exception:
            notes.append("Could not extract logit details from logit_result object.")

    # poisson_result
    if result["poisson"] is None and model_output.get("poisson_result") is not None:
        try:
            res = model_output["poisson_result"]
            coef = float(res.params["HasChildren"]) if "HasChildren" in res.params.index else None
            se = float(res.bse["HasChildren"]) if hasattr(res, "bse") and "HasChildren" in res.bse.index else None
            pval = float(res.pvalues["HasChildren"]) if hasattr(res, "pvalues") and "HasChildren" in res.pvalues.index else None
            result["poisson"] = build_stats(coef, se, pval, "poisson")
        except Exception:
            notes.append("Could not extract poisson details from poisson_result object.")

    # 3) If neither model provided estimates, report the modeling failure notes if present
    if result["logit"] is None and result["poisson"] is None:
        # Gather diagnostic notes from model_output if available
        logit_note = model_output.get("_logit_note")
        pois_note = model_output.get("_poisson_note")
        if logit_note:
            notes.append(logit_note)
        if pois_note:
            notes.append(pois_note)

        # Final conclusion: no usable model estimates
        result["conclusion"] = (
            "No estimates available: both logistic and Poisson models could not be fit "
            "due to insufficient complete cases (see notes). Therefore we cannot determine "
            "from these models whether having children decreases engagement in extramarital affairs."
        )
    else:
        # At least one model produced an estimate. Produce a brief conclusion per model.
        conclusions = []
        if result["logit"] is not None:
            coef = result["logit"].get("coef")
            p = result["logit"].get("pvalue")
            expc = result["logit"].get("exp_coef")
            if coef is None:
                conclusions.append("Logistic model: HasChildren estimate not numeric or missing.")
            else:
                signif = ("statistically significant" if (p is not None and p < 0.05) else "not statistically significant")
                conclusions.append(
                    f"Logistic model: coef={coef:.3f}, OR={expc:.3f} (95% CI approx {result['logit'].get('exp_ci_95')}) — {signif}."
                )
        if result["poisson"] is not None:
            coef = result["poisson"].get("coef")
            p = result["poisson"].get("pvalue")
            expc = result["poisson"].get("exp_coef")
            if coef is None:
                conclusions.append("Poisson model: HasChildren estimate not numeric or missing.")
            else:
                signif = ("statistically significant" if (p is not None and p < 0.05) else "not statistically significant")
                conclusions.append(
                    f"Poisson model: coef={coef:.3f}, IRR={expc:.3f} (95% CI approx {result['poisson'].get('exp_ci_95')}) — {signif}."
                )
        result["conclusion"] = " ".join(conclusions)

    # Attach any gathered notes to the description
    description_lines = []
    description_lines.append(
        "This output summarizes the estimated effect of HasChildren on extramarital affairs from the provided model output."
    )
    if notes:
        description_lines.append("Notes / diagnostics: " + " | ".join(notes))
    if result["conclusion"]:
        description_lines.append("Conclusion: " + result["conclusion"])

    description = " ".join(description_lines)

    return {"object": result, "description": description}