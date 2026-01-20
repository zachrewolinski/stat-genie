def extract_final_answer(model_output):
    """
    Extracts statistics for the primary predictor 'IsHuman' from the model output
    returned by the modeling function.

    Returns a dict with keys:
      - "object": dict with numeric results:
          - coef: log-odds coefficient for IsHuman (float) or None
          - odds_ratio: exp(coef) (float) or None
          - ci_95: tuple (lower, upper) for odds ratio 95% CI or None
          - p_value: p-value for IsHuman (float) or None
          - significant: boolean indicating p_value < 0.05 (or None if p_value is None)
          - conclusion: brief yes/no conclusion string
      - "description": short explanation of the extracted stats and interpretation
    """
    import numpy as np

    result = {
        "object": None,
        "description": None
    }

    # Basic guards / fallbacks
    if model_output is None:
        result["description"] = "No model_output provided."
        return result

    # Try to get the fitted model results object
    model = model_output.get('model', None)

    # Prepare placeholders
    coef = None
    odds_ratio = None
    ci_odds = None
    p_value = None

    # If a statsmodels results object is available, extract from it
    if model is not None:
        try:
            # params and conf_int are typically available
            params = model.params
            conf = model.conf_int()  # log-odds CI by default

            if 'IsHuman' in params.index:
                coef = float(params['IsHuman'])
                # log-odds CI -> convert to odds ratio CI
                if 'IsHuman' in conf.index:
                    ci_log_lower = float(conf.loc['IsHuman', 0])
                    ci_log_upper = float(conf.loc['IsHuman', 1])
                    ci_odds = (float(np.exp(ci_log_lower)), float(np.exp(ci_log_upper)))
                else:
                    ci_odds = None

                odds_ratio = float(np.exp(coef))

                # p-value: may be present on the results object (robust/clustered results include pvalues)
                try:
                    p_value = float(model.pvalues['IsHuman'])
                except Exception:
                    # fallback: some objects store pvalues differently; try attribute access
                    try:
                        p_value = float(model.pvalues.get('IsHuman', None))
                    except Exception:
                        p_value = None
        except Exception:
            # If extraction from model fails, fall through to try using precomputed odds ratios
            coef = None

    # If odds ratios and CIs were precomputed and included in model_output, use them if needed
    if odds_ratio is None and 'odds_ratios' in model_output:
        ors = model_output.get('odds_ratios')
        try:
            if hasattr(ors, 'get'):
                odds_ratio = float(ors.get('IsHuman', None))
            else:
                odds_ratio = float(ors['IsHuman'])
        except Exception:
            odds_ratio = None

    if ci_odds is None and 'odds_ratio_ci' in model_output:
        ci = model_output.get('odds_ratio_ci')
        try:
            # Expect a 2-column structure indexed by parameter names
            if hasattr(ci, 'loc'):
                lower = float(ci.loc['IsHuman', 0])
                upper = float(ci.loc['IsHuman', 1])
                ci_odds = (lower, upper)
            else:
                # If it's a dict-like
                val = ci.get('IsHuman', None)
                if val is not None and len(val) == 2:
                    ci_odds = (float(val[0]), float(val[1]))
        except Exception:
            ci_odds = None

    # If coef is available but odds_ratio not, compute it
    if coef is not None and odds_ratio is None:
        odds_ratio = float(np.exp(coef))

    # Determine significance
    significant = None
    if p_value is not None:
        significant = (p_value < 0.05)

    # Formulate conclusion based on odds ratio and p-value if available
    conclusion = "Unable to determine effect of IsHuman (statistics not available)."
    if odds_ratio is not None:
        if p_value is not None:
            if significant and odds_ratio > 1:
                conclusion = ("Yes: Modern humans (IsHuman) have significantly higher odds of AMTL "
                              f"than non-human primates (OR = {odds_ratio:.3f}, 95% CI = [{ci_odds[0]:.3f}, {ci_odds[1]:.3f}]"
                              f", p = {p_value:.3g}).")
            elif significant and odds_ratio < 1:
                conclusion = ("Yes (but in the opposite direction): Modern humans have significantly lower odds of AMTL "
                              f"(OR = {odds_ratio:.3f}, 95% CI = [{ci_odds[0]:.3f}, {ci_odds[1]:.3f}], p = {p_value:.3g}).")
            else:
                conclusion = (f"No statistically significant difference in AMTL for modern humans (IsHuman) "
                              f"(OR = {odds_ratio:.3f}, 95% CI = [{ci_odds[0]:.3f}, {ci_odds[1]:.3f}]"
                              f", p = {p_value:.3g}).")
        else:
            # p-value not available; base wording on magnitude and CI if available
            if ci_odds is not None:
                # If CI does not contain 1, infer significance
                if ci_odds[0] > 1:
                    conclusion = (f"Evidence suggests modern humans have higher odds of AMTL (OR = {odds_ratio:.3f}, "
                                  f"95% CI = [{ci_odds[0]:.3f}, {ci_odds[1]:.3f}]).")
                elif ci_odds[1] < 1:
                    conclusion = (f"Evidence suggests modern humans have lower odds of AMTL (OR = {odds_ratio:.3f}, "
                                  f"95% CI = [{ci_odds[0]:.3f}, {ci_odds[1]:.3f}]).")
                else:
                    conclusion = (f"No clear evidence of difference (OR = {odds_ratio:.3f}, "
                                  f"95% CI = [{ci_odds[0]:.3f}, {ci_odds[1]:.3f}]).")
            else:
                conclusion = (f"Estimated OR for IsHuman = {odds_ratio:.3f} (no CI or p-value available).")

    # Build returned object
    obj = {
        "coef_log_odds": float(coef) if coef is not None else None,
        "odds_ratio": float(odds_ratio) if odds_ratio is not None else None,
        "odds_ratio_95ci": (float(ci_odds[0]), float(ci_odds[1])) if ci_odds is not None else None,
        "p_value": float(p_value) if p_value is not None else None,
        "significant": bool(significant) if significant is not None else None,
        "alpha": 0.05,
        "conclusion": conclusion
    }

    result["object"] = obj
    # Brief description
    result["description"] = ("Extracted model estimate for the binary predictor 'IsHuman'. "
                             "Key values returned: log-odds coefficient, odds ratio, 95% CI for the odds ratio, "
                             "p-value, and a concise conclusion about whether modern humans show higher AMTL "
                             "after adjusting for age, sex, and tooth class (clustered by population).")
    return result