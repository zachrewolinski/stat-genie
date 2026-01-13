def extract_final_answer(model_output):
    """
    Extracts the effect estimate for the primary IV (MasFem) from a modeling output dict.
    Returns a dict with keys:
      - "object": extracted numeric results (coef, p-value, conf int, transformed effect for GLM)
      - "description": human-readable explanation of what was extracted / why not available

    The function handles:
      - statsmodels OLS results (key 'ols')
      - statsmodels GLM results (keys 'neg_binomial' or 'poisson_fallback')
      - cases where models failed (returns error messages present in the model_output)
    """
    import numpy as np

    out = {"object": None, "description": ""}

    def extract_from_model(model, param_name, model_type):
        # Attempt to pull coefficient, p-value, and 95% CI for param_name
        try:
            params = model.params
        except Exception:
            return None, f"Model object of type {model_type} has no .params attribute."

        # Find a matching parameter name (exact or fallback)
        if param_name in params.index:
            pname = param_name
        else:
            # try common alternative names
            alternatives = [param_name, "MTurkMasFem", "FemaleName"]
            pname = None
            for alt in alternatives:
                if alt in params.index:
                    pname = alt
                    break
            if pname is None:
                return None, f"Parameter '{param_name}' not found in model parameters: available params = {list(params.index)}"

        coef = float(params[pname])

        # p-value
        pval = None
        try:
            pval = float(model.pvalues[pname])
        except Exception:
            pval = None

        # confidence interval
        ci = None
        try:
            ci_df = model.conf_int()
            # conf_int may have labeled index or integer index
            if pname in ci_df.index:
                ci = [float(ci_df.loc[pname, 0]), float(ci_df.loc[pname, 1])]
            else:
                # fallback: try to find by position
                idx = list(params.index).index(pname)
                ci = [float(ci_df.iloc[idx, 0]), float(ci_df.iloc[idx, 1])]
        except Exception:
            ci = None

        result = {
            "param": pname,
            "coef": coef,
            "p_value": pval,
            "conf_int_95": ci,
            "model_type": model_type
        }

        # For count models (GLM with log link), provide exponentiated effect (IRR)
        try:
            fam = getattr(model, "family", None)
            link = None
            if fam is not None:
                # statsmodels families have .link attribute
                link = getattr(fam, "link", None)
            if model_type in ("neg_binomial", "poisson") or (link is not None and hasattr(link, "name") and "log" in str(link).lower()):
                irr = float(np.exp(coef))
                result["exp_coef"] = irr
                if ci is not None:
                    result["exp_conf_int_95"] = [float(np.exp(ci[0])), float(np.exp(ci[1]))]
        except Exception:
            # ignore any transformation errors
            pass

        # Short interpretation string
        interp = f"Extracted parameter '{pname}' from {model_type}. "
        interp += f"Coefficient = {coef:.4g}"
        if pval is not None:
            interp += f", p-value = {pval:.4g}"
        if ci is not None:
            interp += f", 95% CI = [{ci[0]:.4g}, {ci[1]:.4g}]"
        if "exp_coef" in result:
            interp += f". Exponentiated effect = {result['exp_coef']:.4g} (interpretable as multiplicative effect on counts)."

        return result, interp

    # 1) Prefer OLS result if present
    if isinstance(model_output, dict):
        if "ols" in model_output and model_output["ols"] is not None:
            res, desc = extract_from_model(model_output["ols"], "MasFem", "OLS")
            out["object"] = res
            out["description"] = desc
            return out

        # 2) Else prefer negative binomial
        if "neg_binomial" in model_output and model_output["neg_binomial"] is not None:
            res, desc = extract_from_model(model_output["neg_binomial"], "MasFem", "neg_binomial")
            out["object"] = res
            out["description"] = desc
            return out

        # 3) Else Poisson fallback
        if "poisson_fallback" in model_output and model_output["poisson_fallback"] is not None:
            res, desc = extract_from_model(model_output["poisson_fallback"], "MasFem", "poisson")
            out["object"] = res
            out["description"] = desc
            return out

        # 4) No fitted models present: collate error messages if available
        msgs = []
        for key in ["ols_error", "neg_binomial_error", "neg_binomial", "poisson_fallback", "ols"]:
            if key in model_output and model_output[key] is not None:
                # include string errors or repr of object
                val = model_output[key]
                try:
                    if isinstance(val, str):
                        msgs.append(f"{key}: {val}")
                    else:
                        msgs.append(f"{key}: {repr(val)}")
                except Exception:
                    msgs.append(f"{key}: <unrepresentable>")

        if not msgs:
            out["description"] = "No model results or error messages found in model_output."
        else:
            out["description"] = "No fitted model available to extract 'MasFem' effect. Errors / notes from model fitting: " + " | ".join(msgs)

        out["object"] = None
        return out

    # If model_output is not a dict
    out["description"] = "model_output is not a dictionary in the expected format."
    out["object"] = None
    return out

# Example behavior on the provided model_output:
# The function will return object = None and description containing the ols_error and neg_binomial_error strings,
# indicating that 'FemaleName' was undefined and 'Deaths' column was missing.