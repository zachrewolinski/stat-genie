def extract_final_answer(model_output):
    """
    Extracts coefficient, SE, p-value, 95% CI, and multiplicative effect (percent change)
    for predictors of interest from a statsmodels MixedLM results object.

    Returns:
      {
        "object": {
           "age": { "coef": ..., "se": ..., "p": ..., "ci_lower": ..., "ci_upper": ..., "pct_change": ... },
           "sex_M": { ... },
           "Help": { ... }
        },
        "description": "Concise interpretation of each predictor in context"
      }
    """
    import numpy as np

    # Predictors of interest
    preds = ['age', 'sex_M', 'Help']

    # Prepare containers
    result_obj = {}
    notes = []

    # Helper functions
    def has_key(container, key):
        if container is None:
            return False
        try:
            if hasattr(container, "index"):
                return key in container.index
        except Exception:
            pass
        try:
            if hasattr(container, "keys"):
                return key in container.keys()
        except Exception:
            pass
        try:
            return key in container
        except Exception:
            return False

    def get_value(container, key):
        if container is None:
            return None
        try:
            return container.loc[key]
        except Exception:
            pass
        try:
            return container[key]
        except Exception:
            pass
        return None

    # Try to obtain fixed-effect parameter estimates and related statistics robustly
    try:
        # Prefer fe_params if available
        if hasattr(model_output, 'fe_params'):
            fe = model_output.fe_params
        else:
            # fallback: use model.exog_names to select fixed-effect params from params
            exog_names = getattr(getattr(model_output, "model", None), 'exog_names', None)
            if exog_names is not None and hasattr(model_output, "params"):
                try:
                    fe = model_output.params.loc[exog_names]
                except Exception:
                    # if that fails, just take params
                    fe = model_output.params
            else:
                fe = getattr(model_output, "params", None)

        # Standard errors for fixed effects
        if hasattr(model_output, 'bse_fe'):
            bse = model_output.bse_fe
        else:
            # fallback: use bse and restrict to fe index if possible
            if hasattr(model_output, 'bse'):
                bse = model_output.bse
                try:
                    if hasattr(fe, "index"):
                        bse = bse.loc[fe.index]
                except Exception:
                    pass
            else:
                bse = None

        # p-values (may include both fixed and other params)
        if hasattr(model_output, 'pvalues'):
            pvals = model_output.pvalues
            try:
                if hasattr(fe, "index"):
                    pvals = pvals.loc[fe.index]
            except Exception:
                pass
        else:
            pvals = None

        # Confidence intervals
        try:
            ci_df = model_output.conf_int()
        except Exception:
            ci_df = None

    except Exception as e:
        raise RuntimeError(f"Unable to extract model statistics: {e}")

    # Extract stats for each predictor and build interpretation
    for pred in preds:
        entry = {"coef": None, "se": None, "p": None, "ci_lower": None, "ci_upper": None, "pct_change": None}

        present = has_key(fe, pred)
        if present:
            # Coefficient
            try:
                coef_val = get_value(fe, pred)
                entry["coef"] = float(coef_val) if coef_val is not None else None
            except Exception:
                entry["coef"] = None

            # SE
            try:
                se_val = get_value(bse, pred)
                entry["se"] = float(se_val) if se_val is not None else None
            except Exception:
                entry["se"] = None

            # p-value
            try:
                p_val = get_value(pvals, pred)
                entry["p"] = float(p_val) if p_val is not None else None
            except Exception:
                entry["p"] = None

            # CI
            if ci_df is not None:
                try:
                    ci_row = ci_df.loc[pred].values
                    entry["ci_lower"] = float(ci_row[0])
                    entry["ci_upper"] = float(ci_row[1])
                except Exception:
                    entry["ci_lower"] = None
                    entry["ci_upper"] = None

            # Percent change on original rate scale: exp(coef)-1
            try:
                entry["pct_change"] = float((np.exp(entry["coef"]) - 1.0) * 100.0) if entry["coef"] is not None else None
            except Exception:
                entry["pct_change"] = None

            # Build interpretation sentence for this predictor
            if entry["coef"] is not None:
                sig = None
                if entry["p"] is not None:
                    sig = entry["p"] < 0.05
                # Use CI to check if excludes zero if p not available
                if sig is None and entry["ci_lower"] is not None and entry["ci_upper"] is not None:
                    sig = not (entry["ci_lower"] <= 0 <= entry["ci_upper"])
                sign_text = "increase" if entry["coef"] > 0 else ("decrease" if entry["coef"] < 0 else "no change")
                sig_text = "statistically significant" if sig else "not statistically significant"

                # Formatting strings safely
                coef_str = f"{entry['coef']:.3f}"
                se_str = f"{entry['se']:.3f}" if entry['se'] is not None else "NA"
                p_str = f"{entry['p']:.3f}" if entry['p'] is not None else "NA"
                ci_lower_str = f"{entry['ci_lower']:.3f}" if entry['ci_lower'] is not None else "NA"
                ci_upper_str = f"{entry['ci_upper']:.3f}" if entry['ci_upper'] is not None else "NA"
                pct_str = f"{entry['pct_change']:.1f}%" if entry['pct_change'] is not None else "NA"

                # Specific textual interpretation (LogRate is log((nuts_opened + 0.5)/seconds))
                if pred == 'age':
                    var_label = "Age (years)"
                    detail = f"Each additional year is associated with a {pct_str} {sign_text} in the nuts-opened-per-second rate (on average)."
                elif pred == 'sex_M':
                    var_label = "Sex (male vs female)"
                    detail = f"Being male (vs female) is associated with a {pct_str} {sign_text} in the nuts-opened-per-second rate (on average)."
                elif pred == 'Help':
                    var_label = "Received help (yes vs no)"
                    detail = f"Receiving help (vs not) is associated with a {pct_str} {sign_text} in the nuts-opened-per-second rate (on average)."
                else:
                    var_label = pred
                    detail = f"Effect: {coef_str} on log-rate ({sig_text})."

                sentence = (
                    f"{var_label}: coef={coef_str}, SE={se_str}, p={p_str}, 95%CI=[{ci_lower_str}, {ci_upper_str}]. "
                    f"{detail} ({sig_text})."
                )
            else:
                sentence = f"{pred}: parameter found but coefficient could not be extracted."

            notes.append(sentence)
        else:
            notes.append(f"{pred}: not present among model fixed-effect parameters.")

        result_obj[pred] = entry

    # Compose a concise description summarizing results
    description = (
        "Extracted fixed-effect estimates for predictors of interest. "
        "Estimates are on the log-rate scale where LogRate = log((nuts_opened + 0.5) / seconds). "
        "For each predictor we report coefficient, standard error (SE), p-value, 95% CI, and "
        "the approximate percent change in the raw nuts-per-second rate (exp(coef)-1). "
        "Summary by predictor:\n- " + "\n- ".join(notes)
    )

    return {"object": result_obj, "description": description}