def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, t-stats, p-values, and 95% CIs for the
    beauty terms from the two fitted statsmodels results objects in model_output.

    Returns:
      {
        "object": {
          "continuous": {
            "term": "Beauty_z",
            "coef": ...,
            "se": ...,
            "t": ...,
            "p": ...,
            "ci_lower": ...,
            "ci_upper": ...,
            "nobs": ...
          },
          "binary": {
            "term": "BeautyBinary",
            ...
          }
        },
        "description": "<brief interpretation and yes/no conclusion about effect>"
      }
    """
    import pandas as pd
    import numpy as np

    result = {"continuous": None, "binary": None}
    descriptions = []

    # mapping from keys in model_output to variable names
    specs = [
        ("model_beauty_continuous", "Beauty_z", "continuous"),
        ("model_beauty_binary", "BeautyBinary", "binary"),
    ]

    for model_key, varname, out_key in specs:
        res = model_output.get(model_key)
        if res is None:
            result[out_key] = {"error": f"Model object '{model_key}' not found in model_output"}
            descriptions.append(f"Model '{model_key}': not available.")
            continue

        # Initialize container for this model's extracted stats
        stats = {"term": varname}

        # Safe extraction helpers
        try:
            stats["coef"] = float(res.params[varname])
        except Exception:
            stats["coef"] = None

        try:
            stats["se"] = float(res.bse[varname])
        except Exception:
            stats["se"] = None

        try:
            # statsmodels exposes tvalues and pvalues on the result
            stats["t"] = float(res.tvalues[varname])
        except Exception:
            stats["t"] = None

        try:
            stats["p"] = float(res.pvalues[varname])
        except Exception:
            stats["p"] = None

        # Confidence intervals (95%)
        try:
            conf = res.conf_int()
            # conf can be DataFrame or ndarray; handle both
            if isinstance(conf, pd.DataFrame):
                ci_vals = conf.loc[varname].values
            else:
                # conf is ndarray; find index of varname in model exog names
                names = list(res.model.exog_names)
                idx = names.index(varname)
                ci_vals = conf[idx]
            stats["ci_lower"] = float(ci_vals[0])
            stats["ci_upper"] = float(ci_vals[1])
        except Exception:
            stats["ci_lower"] = stats["ci_upper"] = None

        # Number of observations if available
        try:
            stats["nobs"] = int(res.nobs)
        except Exception:
            stats["nobs"] = None

        # Store
        result[out_key] = stats

        # Build a brief textual interpretation for this term
        if stats["coef"] is None:
            descriptions.append(f"{model_key}: Could not extract coefficient for {varname}.")
            continue

        # Direction and significance
        direction = "positive" if stats["coef"] > 0 else ("negative" if stats["coef"] < 0 else "no direction (coef=0)")
        sig = None
        if stats["p"] is None:
            sig = "p-value unavailable"
        else:
            sig = "statistically significant (p < 0.05)" if stats["p"] < 0.05 else f"not statistically significant (p = {stats['p']:.3f})"

        # Plain-language effect size
        if out_key == "continuous":
            # coefficient interpreted as change in EvalScore per 1 SD increase in beauty
            effect_text = (
                f"One SD increase in beauty is associated with a {stats['coef']:.3f}-point change "
                f"in evaluation score ({direction}, {sig}). 95% CI [{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]."
            )
        else:
            # coefficient interpreted as difference in EvalScore for 'beautiful' vs 'not'
            effect_text = (
                f"Being coded 'beautiful' is associated with a {stats['coef']:.3f}-point difference "
                f"in evaluation score ({direction}, {sig}). 95% CI [{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]."
            )

        descriptions.append(f"{model_key} ({varname}): {effect_text}")

    # Decide overall short answer (yes/no) based on significance in either specification.
    cont_p = result["continuous"].get("p") if isinstance(result["continuous"], dict) else None
    bin_p = result["binary"].get("p") if isinstance(result["binary"], dict) else None

    overall = None
    if (cont_p is not None and cont_p < 0.05) or (bin_p is not None and bin_p < 0.05):
        overall = "Yes — there is evidence that instructor appearance (beauty) is associated with teaching evaluation scores in at least one specification."
    else:
        overall = "No — there is no clear evidence that instructor appearance (beauty) is associated with teaching evaluation scores at conventional significance levels in either specification."

    # Combine descriptions
    description_text = (
        overall + " Detailed results:\n" + "\n".join(descriptions)
        + "\n\nNotes: For the continuous model, the coefficient is the change in EvalScore per 1 SD increase in beauty. "
        "For the binary model, the coefficient is the difference in EvalScore between instructors coded as 'beautiful' vs not. "
        "All statistics are taken from the fitted models (cluster-robust SEs if provided by the fit)."
    )

    return {"object": result, "description": description_text}