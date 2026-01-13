def extract_final_answer(model_output):
    """
    Extracts key statistics from a fitted GLM model output (as returned by the provided `model` function).
    Returns a dictionary with:
      - "object": a dict containing coefficient table (coef, se, p, confint),
                  exponentiated coefficients (IRRs) and CIs, average predicted rate,
                  and predicted rates per hour for representative groups (livebait x camper).
      - "description": a short human-readable interpretation of the most relevant results
                       (average predicted fish/hour, effect of livebait and camper, effect per additional person).
    """
    import numpy as np
    import pandas as pd

    # Basic checks
    if not isinstance(model_output, dict) or "glm_results" not in model_output:
        raise ValueError("model_output must be a dict containing 'glm_results' (statsmodels GLMResultsWrapper).")

    results = model_output["glm_results"]

    # Extract parameter estimates, p-values, conf intervals
    params = results.params.copy()
    bse = results.bse.copy() if hasattr(results, "bse") else None
    pvalues = results.pvalues.copy() if hasattr(results, "pvalues") else None
    try:
        conf = results.conf_int().copy()
        conf.columns = ["conf_low", "conf_high"]
    except Exception:
        # fallback: construct approximate CIs using normal approx if conf_int fails
        if bse is not None:
            z = 1.96
            conf = pd.DataFrame({
                "conf_low": params - z * bse,
                "conf_high": params + z * bse
            })
        else:
            conf = pd.DataFrame({
                "conf_low": params * np.nan,
                "conf_high": params * np.nan
            })

    coef_table = pd.DataFrame({
        "coef": params,
        "se": bse,
        "pvalue": pvalues,
        "conf_low": conf["conf_low"],
        "conf_high": conf["conf_high"]
    })

    # Exponentiate coefficients to get incidence rate ratios (IRR) and their CIs
    irr = np.exp(params)
    irr_conf_low = np.exp(conf["conf_low"])
    irr_conf_high = np.exp(conf["conf_high"])
    irr_table = pd.DataFrame({
        "IRR": irr,
        "IRR_conf_low": irr_conf_low,
        "IRR_conf_high": irr_conf_high
    })

    # Prepare a concise results for key predictors
    key_predictors = ["livebait", "camper", "group_size"]
    key_summary = {}
    for k in key_predictors:
        if k in coef_table.index:
            key_summary[k] = {
                "coef": float(coef_table.loc[k, "coef"]),
                "se": float(coef_table.loc[k, "se"]) if pd.notnull(coef_table.loc[k, "se"]) else None,
                "pvalue": float(coef_table.loc[k, "pvalue"]) if pd.notnull(coef_table.loc[k, "pvalue"]) else None,
                "conf_low": float(coef_table.loc[k, "conf_low"]) if pd.notnull(coef_table.loc[k, "conf_low"]) else None,
                "conf_high": float(coef_table.loc[k, "conf_high"]) if pd.notnull(coef_table.loc[k, "conf_high"]) else None,
                "IRR": float(irr_table.loc[k, "IRR"]),
                "IRR_conf_low": float(irr_table.loc[k, "IRR_conf_low"]),
                "IRR_conf_high": float(irr_table.loc[k, "IRR_conf_high"])
            }
        else:
            key_summary[k] = None

    # Average predicted rate per hour (if predictions were returned by the model function)
    avg_pred_rate_per_hour = None
    if "predictions" in model_output and isinstance(model_output["predictions"], (pd.DataFrame, dict)):
        preds = model_output["predictions"]
        try:
            avg_pred_rate_per_hour = float(preds["predicted_rate_per_hour"].mean())
        except Exception:
            avg_pred_rate_per_hour = None

    # Predict expected fish-per-hour for representative combinations of livebait and camper,
    # holding group_size at its observed mean. We predict for 1 hour (offset = log(1) = 0),
    # so predicted count == predicted rate per hour.
    predicted_rates_by_group = {}
    try:
        exog_names = list(results.model.exog_names)
        # compute mean group_size from the exog matrix used to fit the model
        if "group_size" in exog_names:
            gs_idx = exog_names.index("group_size")
            group_size_mean = float(results.model.exog[:, gs_idx].mean())
        else:
            group_size_mean = 0.0

        # build prediction dataframe for 4 combos
        combos = []
        for lb in [0, 1]:
            for camp in [0, 1]:
                row = {}
                for name in exog_names:
                    if name == "const":
                        row[name] = 1.0
                    elif name == "livebait":
                        row[name] = float(lb)
                    elif name == "camper":
                        row[name] = float(camp)
                    elif name == "group_size":
                        row[name] = float(group_size_mean)
                    else:
                        # If there are other columns present (unexpected), set to zero
                        row[name] = 0.0
                combos.append(row)
        exog_pred = pd.DataFrame(combos, columns=exog_names)

        # offset = log(1) = 0 for 1-hour predictions
        offset_zero = np.zeros(len(exog_pred))
        pred_counts_for_1hr = results.predict(exog_pred, offset=offset_zero)
        # attach to dictionary with readable keys
        i = 0
        for lb in [0, 1]:
            for camp in [0, 1]:
                key = f"livebait={lb}_camper={camp}"
                predicted_rates_by_group[key] = float(pred_counts_for_1hr[i])  # expected fish per hour
                i += 1
    except Exception:
        predicted_rates_by_group = None

    # Build the object to return
    object_to_return = {
        "coef_table": coef_table,                     # full coef table (pandas DataFrame)
        "irr_table": irr_table,                       # exponentiated coefficients and CIs (pandas DataFrame)
        "key_summary": key_summary,                   # compact summary for livebait, camper, group_size (dict)
        "avg_pred_rate_per_hour": avg_pred_rate_per_hour,  # float or None
        "predicted_rates_by_group": predicted_rates_by_group  # dict of predicted fish/hour for combos (or None)
    }

    # Short human-readable description / interpretation
    # We include the main takeaways: average predicted rate (if available) and the multiplicative effects (IRRs).
    description_lines = []
    if avg_pred_rate_per_hour is not None:
        description_lines.append(f"Average model-predicted catch rate across observed visits: {avg_pred_rate_per_hour:.3f} fish/hour.")
    else:
        description_lines.append("Average model-predicted catch rate per hour not available from provided predictions.")

    # Interpret livebait
    if key_summary.get("livebait") is not None:
        ks = key_summary["livebait"]
        description_lines.append(
            f"Using live bait is associated with an IRR = {ks['IRR']:.3f} "
            f"(95% CI: {ks['IRR_conf_low']:.3f}–{ks['IRR_conf_high']:.3f}), p = {ks['pvalue']:.3g}. "
            "An IRR > 1 indicates higher expected fish-per-hour when live bait is used."
        )
    else:
        description_lines.append("Livebait coefficient not available in the fitted model output.")

    # Interpret camper
    if key_summary.get("camper") is not None:
        ks = key_summary["camper"]
        description_lines.append(
            f"Having a camper is associated with an IRR = {ks['IRR']:.3f} "
            f"(95% CI: {ks['IRR_conf_low']:.3f}–{ks['IRR_conf_high']:.3f}), p = {ks['pvalue']:.3g}."
        )
    else:
        description_lines.append("Camper coefficient not available in the fitted model output.")

    # Interpret group_size (per additional person)
    if key_summary.get("group_size") is not None:
        ks = key_summary["group_size"]
        description_lines.append(
            f"Each additional person in the group multiplies expected catch rate by IRR = {ks['IRR']:.3f} "
            f"(95% CI: {ks['IRR_conf_low']:.3f}–{ks['IRR_conf_high']:.3f}), p = {ks['pvalue']:.3g}."
        )
    else:
        description_lines.append("Group_size coefficient not available in the fitted model output.")

    if predicted_rates_by_group is not None:
        description_lines.append("Representative predicted fish/hour for 1-hour trip (group_size held at observed mean):")
        for k, v in predicted_rates_by_group.items():
            description_lines.append(f"  {k}: {v:.3f} fish/hour")

    description = " ".join(description_lines)

    return {
        "object": object_to_return,
        "description": description
    }