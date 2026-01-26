def extract_final_answer(model_output):
    """
    Extracts statistics for the 'female' predictor from a fitted logit model output dict
    and returns a compact result plus a short interpretation.

    Returns:
        dict with keys:
            - "object": dict of extracted numeric results:
                - coef: log-odds coefficient for female
                - coef_ci: 95% CI for coef (tuple: lower, upper)
                - pvalue: p-value for female coefficient
                - odds_ratio: exp(coef)
                - odds_ratio_ci: 95% CI for odds ratio (tuple: lower, upper)
                - significant: boolean (p < 0.05)
                - pct_change_in_odds: (odds_ratio - 1) * 100
            - "description": short plain-language interpretation in the task context
    """
    import math

    # Helper to safely get item from pandas-like objects or plain dicts/Series
    def _get_item(obj, key):
        if obj is None:
            return None
        try:
            # pandas Series / DataFrame .loc
            return obj.loc[key]
        except Exception:
            try:
                return obj[key]
            except Exception:
                return None

    # Attempt to extract from fitted model if available
    fit = model_output.get('model_fit')
    odds_ratio = None
    odds_ci = None
    coef = None
    coef_ci = None
    pvalue = None

    try:
        if fit is not None:
            # Prefer reading from the model_fit object (statsmodels results)
            coef = float(fit.params['female'])
            pvalue = float(fit.pvalues['female'])
            # conf_int returns a DataFrame with two columns; index is variable names
            conf = fit.conf_int().loc['female']
            coef_ci = (float(conf.iloc[0]), float(conf.iloc[1]))
            odds_ratio = math.exp(coef)
            odds_ci = (math.exp(coef_ci[0]), math.exp(coef_ci[1]))
        else:
            # Fall back to precomputed items in the model_output dict
            # odds_ratios and conf_int_exp are expected keys from the modeling function
            odds_series = model_output.get('odds_ratios')
            conf_exp = model_output.get('conf_int_exp')
            pseries = model_output.get('pvalues')

            odds_val = _get_item(odds_series, 'female')
            if odds_val is not None:
                odds_ratio = float(odds_val)

            conf_val = _get_item(conf_exp, 'female')
            if conf_val is not None:
                # conf_exp expected to be a two-column structure (lower, upper)
                try:
                    odds_ci = (float(conf_val.iloc[0]), float(conf_val.iloc[1]))
                except Exception:
                    # In case conf_val is a plain iterable
                    odds_ci = (float(conf_val[0]), float(conf_val[1]))

            if odds_ratio is not None and odds_ci is not None:
                coef = math.log(odds_ratio)
                coef_ci = (math.log(odds_ci[0]), math.log(odds_ci[1]))

            pval = _get_item(pseries, 'female')
            if pval is not None:
                pvalue = float(pval)
    except Exception as e:
        # If any extraction step fails, raise a clear error for debugging
        raise RuntimeError(f"Failed to extract 'female' stats from model_output: {e}")

    # Build result object
    if coef is None and odds_ratio is None:
        raise RuntimeError("Could not find 'female' coefficient/odds ratio in model_output.")

    result_obj = {
        "coef": coef,
        "coef_ci": coef_ci,
        "pvalue": pvalue,
        "odds_ratio": odds_ratio,
        "odds_ratio_ci": odds_ci,
        "significant": (pvalue is not None and pvalue < 0.05),
        "pct_change_in_odds": (odds_ratio - 1) * 100 if odds_ratio is not None else None
    }

    # Create a concise interpretation/description
    if result_obj["significant"]:
        desc = (
            f"Yes — being female is associated with higher odds of mortgage approval "
            f"controlling for the listed covariates. Estimated odds ratio = {odds_ratio:.3f} "
            f"(95% CI: {odds_ci[0]:.3f} to {odds_ci[1]:.3f}), p = {pvalue:.3g}. "
            f"That corresponds to about a {result_obj['pct_change_in_odds']:.1f}% increase in odds "
            f"of approval relative to male applicants. (Log-odds coef = {coef:.3f}, 95% CI: "
            f"{coef_ci[0]:.3f} to {coef_ci[1]:.3f}.)"
        )
    else:
        desc = (
            f"No strong evidence that gender affects mortgage approval after adjustment. "
            f"Estimated odds ratio = {odds_ratio:.3f} "
            f"(95% CI: {odds_ci[0]:.3f} to {odds_ci[1]:.3f}), p = {pvalue:.3g}. "
            f"This does not reach conventional significance (p < 0.05)."
        )

    # Add a brief caveat
    desc += " Results are associative (observational) and do not by themselves establish causation."

    return {"object": result_obj, "description": desc}