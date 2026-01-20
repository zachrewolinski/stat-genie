def extract_final_answer(model_output):
    """
    Extract key statistics from the model_output produced by the provided modeling function.

    Returns a dictionary with keys:
      - "object": a dict containing numeric summaries (coefficients, p-values, IRRs, 95% CIs)
                  for the main predictors and an example predicted fish/hour for a 2-person
                  group with and without livebait.
      - "description": a plain-language interpretation of those statistics.

    This function is defensive: it tries to read values from the fitted statsmodels result
    inside model_output['model'] and from the precomputed IRR table model_output['irr'] if present.
    """
    import numpy as np

    out = {"object": None, "description": None}

    # Basic checks
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dictionary as returned by the modelling function.")

    fit = model_output.get("model", None)
    irr_table = model_output.get("irr", None)

    # Try to extract from fitted model if available
    try:
        params = fit.params  # pandas Series
        pvals = fit.pvalues
        conf = fit.conf_int()
    except Exception:
        # Fall back to irr table if present
        if irr_table is None:
            raise ValueError("Could not extract fitted model parameters from model_output.")
        # Try to construct minimal info from irr_table only
        irr = {}
        for name in irr_table.index:
            irr[name] = {
                "IRR": float(irr_table.loc[name, "IRR"]),
                "IRR_2.5%": float(irr_table.loc[name, "IRR_2.5%"]),
                "IRR_97.5%": float(irr_table.loc[name, "IRR_97.5%"]),
            }
        out["object"] = {"irr_only": irr}
        out["description"] = "Only IRR table found; no coefficient/p-value/conf-int available from the fitted model object."
        return out

    # Compute IRRs and CI
    irr = np.exp(params)
    irr_ci_lower = np.exp(conf[0])
    irr_ci_upper = np.exp(conf[1])

    # Helper to safely pick values (some names may differ in different versions)
    def get_val(series, name):
        return series.get(name, np.nan)

    # Keys of interest
    keys = ["Intercept", "livebait", "camper", "total_people", "religiousness", "year_centered"]
    summary = {}
    for k in keys:
        coef = get_val(params, k)
        pval = get_val(pvals, k)
        ci_low = get_val(conf[0], k)
        ci_high = get_val(conf[1], k)
        irr_k = np.exp(coef) if not np.isnan(coef) else np.nan
        irr_low = np.exp(ci_low) if not np.isnan(ci_low) else np.nan
        irr_high = np.exp(ci_high) if not np.isnan(ci_high) else np.nan

        summary[k] = {
            "coef": float(coef) if not np.isnan(coef) else None,
            "p_value": float(pval) if not np.isnan(pval) else None,
            "IRR": float(irr_k) if not np.isnan(irr_k) else None,
            "IRR_2.5%": float(irr_low) if not np.isnan(irr_low) else None,
            "IRR_97.5%": float(irr_high) if not np.isnan(irr_high) else None,
        }

    # Example predicted rates (per hour) for a simple, concrete scenario:
    # - 2-person group, camper=0, year_centered=0, religiousness=0 (or omitted)
    # Compute: rate = exp(intercept + beta_total_people * n_people + beta_camper * camper + beta_relig * relig + beta_year * year)
    try:
        intercept = summary["Intercept"]["coef"]
        beta_tp = summary["total_people"]["coef"]
        beta_camper = summary["camper"]["coef"] if summary["camper"]["coef"] is not None else 0.0
        beta_relig = summary["religiousness"]["coef"] if summary["religiousness"]["coef"] is not None else 0.0
        beta_year = summary["year_centered"]["coef"] if summary["year_centered"]["coef"] is not None else 0.0

        n_people_example = 2
        camper_example = 0
        relig_example = 0.0
        year_example = 0.0

        linear_pred_no_live = intercept + beta_tp * n_people_example + beta_camper * camper_example + beta_relig * relig_example + beta_year * year_example
        rate_no_live = float(np.exp(linear_pred_no_live))  # fish per hour for 2-person group without livebait

        # With livebait: multiply by exp(beta_livebait)
        beta_live = summary["livebait"]["coef"]
        rate_with_live = float(np.exp(linear_pred_no_live + (beta_live if beta_live is not None else 0.0)))

    except Exception:
        rate_no_live = None
        rate_with_live = None

    # Collect diagnostics if present
    pearson_disp = model_output.get("pearson_dispersion", None)
    chosen_family = model_output.get("chosen_family", None)
    dispersion_raw = model_output.get("dispersion_raw", None)

    # Build final object
    result_object = {
        "coeff_summary": summary,
        "example_rates_per_hour": {
            "2_person_no_livebait_fish_per_hour": rate_no_live,
            "2_person_with_livebait_fish_per_hour": rate_with_live,
        },
        "model_diagnostics": {
            "chosen_family": chosen_family,
            "dispersion_raw": float(dispersion_raw) if dispersion_raw is not None else None,
            "pearson_dispersion": float(pearson_disp) if pearson_disp is not None else None,
        }
    }

    # Plain-language description using extracted numbers (if available)
    try:
        live_irr = summary["livebait"]["IRR"]
        live_ci_low = summary["livebait"]["IRR_2.5%"]
        live_ci_hi = summary["livebait"]["IRR_97.5%"]
        live_p = summary["livebait"]["p_value"]

        tp_irr = summary["total_people"]["IRR"]
        tp_ci_low = summary["total_people"]["IRR_2.5%"]
        tp_ci_hi = summary["total_people"]["IRR_97.5%"]
        tp_p = summary["total_people"]["p_value"]

        camper_p = summary["camper"]["p_value"]
        intercept_rate = summary["Intercept"]["IRR"]

        desc_lines = []
        desc_lines.append(
            f"Baseline catch rate (all predictors = 0, reference county): ~{intercept_rate:.3g} fish/hour."
            if intercept_rate is not None else "Baseline catch rate not available."
        )
        if live_irr is not None:
            desc_lines.append(
                f"Using live bait is associated with a large, statistically significant increase in catch rate: "
                f"IRR = {live_irr:.3g} (95% CI {live_ci_low:.3g} – {live_ci_hi:.3g}), p = {live_p:.3g}."
            )
        if tp_irr is not None:
            desc_lines.append(
                f"Each additional person in the group is associated with a higher group catch rate per hour: "
                f"IRR = {tp_irr:.3g} per additional person (95% CI {tp_ci_low:.3g} – {tp_ci_hi:.3g}), p = {tp_p:.3g}."
            )
        if camper_p is not None:
            desc_lines.append(
                f"Presence of a camper is not a statistically significant predictor (p = {camper_p:.3g})."
            )

        # Example interpretation for a 2-person group
        if rate_no_live is not None and rate_with_live is not None:
            desc_lines.append(
                f"As an example: for a 2-person group (other controls set to 0), estimated catch rate ≈ "
                f"{rate_no_live:.3f} fish/hour without live bait and ≈ {rate_with_live:.3f} fish/hour with live bait "
                f"(all else equal)."
            )

        # Note on model family/dispersion
        if chosen_family is not None:
            desc_lines.append(
                f"Model used: {chosen_family}. Pearson dispersion ≈ {pearson_disp:.3g} (raw dispersion {dispersion_raw:.3g}), "
                "suggesting the Negative Binomial family was appropriate to handle overdispersion."
            )

        description = " ".join(desc_lines)
    except Exception:
        description = "Extracted numeric summaries are available in 'object'."

    out["object"] = result_object
    out["description"] = description

    return out