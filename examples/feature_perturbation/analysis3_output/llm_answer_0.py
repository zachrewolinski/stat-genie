def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, confidence intervals, nobs, R-squared,
    and a short interpretation for the primary independent variables in the provided
    statsmodels RegressionResultsWrapper objects.

    Expects model_output to be a dict-like object with keys:
      - 'model_name'   : model where primary IV is 'name_c' (continuous femininity)
      - 'model_binary' : model where primary IV is 'elapsedyrs' (binary female-name indicator)
    Returns a dictionary with keys:
      - "object": a nested dict with numeric statistics for each model and IV
      - "description": a short natural-language interpretation of whether the results
                       support the hypothesis that more feminine names lead to more deaths
                       (i.e., positive & statistically significant association).
    """
    import numpy as np

    # Helper to safely extract stats for a variable from a statsmodels results object
    def get_stats(model, var):
        if model is None:
            return None
        out = {}
        try:
            params = model.params
            pvalues = model.pvalues
            bse = model.bse
            tvalues = model.tvalues
            conf = model.conf_int()  # default 95%
            nobs = int(model.nobs) if hasattr(model, "nobs") else None
            rsq = float(model.rsquared) if hasattr(model, "rsquared") else None
        except Exception as e:
            raise RuntimeError(f"Failed to extract standard attributes from model: {e}")

        if var not in params.index:
            raise KeyError(f"Variable '{var}' not found in model parameters. Available: {list(params.index)}")

        beta = float(params[var])
        se = float(bse[var]) if var in bse.index else None
        t = float(tvalues[var]) if var in tvalues.index else None
        p = float(pvalues[var]) if var in pvalues.index else None
        ci_low, ci_high = (float(conf.loc[var, 0]), float(conf.loc[var, 1])) if var in conf.index else (None, None)

        # Interpret effect on original count scale: outcome is log(ndam15 + 1).
        # A one-unit increase in IV multiplies (ndam15 + 1) by exp(beta).
        # Percent change approx = (exp(beta) - 1) * 100.
        try:
            pct_effect = (np.exp(beta) - 1) * 100.0
        except Exception:
            pct_effect = None

        out.update({
            "variable": var,
            "beta": beta,
            "std_error": se,
            "t_value": t,
            "p_value": p,
            "ci_95": [ci_low, ci_high],
            "n_obs": nobs,
            "r_squared": rsq,
            "pct_effect_on_ndam15_plus1": pct_effect,
            "interpretation_note": (
                "Dependent variable is log(ndam15 + 1). "
                "So exp(beta)-1 gives the multiplicative percent change in (ndam15 + 1) "
                "for a one-unit increase in the IV."
            )
        })
        return out

    # Prepare output structure
    results_obj = {}
    descriptions = []

    # Accept both dict-like model_output and a single statsmodels object (less likely)
    model_name = None
    model_binary = None
    if isinstance(model_output, dict):
        model_name = model_output.get('model_name')
        model_binary = model_output.get('model_binary')
    else:
        # If a single model was passed, assume it's the continuous-name model
        model_name = model_output

    # Extract stats for continuous name femininity (name_c) if model present
    if model_name is not None:
        try:
            stats_name = get_stats(model_name, 'name_c')
            results_obj['model_name'] = stats_name
            # Decision rule for hypothesis:
            if stats_name["p_value"] is not None and stats_name["p_value"] < 0.05 and stats_name["beta"] > 0:
                descriptions.append(
                    "model_name: 'name_c' has a positive, statistically significant coefficient (p < 0.05). "
                    "This is consistent with the hypothesis that more feminine hurricane names are associated with "
                    "greater harm (higher deaths), i.e., fewer precautions taken."
                )
            elif stats_name["p_value"] is not None and stats_name["p_value"] < 0.05 and stats_name["beta"] <= 0:
                descriptions.append(
                    "model_name: 'name_c' has a statistically significant coefficient (p < 0.05) but the sign is "
                    "non-positive (beta <= 0), which does NOT support the hypothesis."
                )
            else:
                descriptions.append(
                    "model_name: 'name_c' is not statistically significant (p >= 0.05). No reliable evidence from "
                    "this model that continuous femininity of the name affects log(deaths)."
                )
        except Exception as e:
            results_obj['model_name_error'] = str(e)
            descriptions.append(f"model_name: could not extract stats: {e}")

    # Extract stats for binary female-name indicator (elapsedyrs) if model present
    if model_binary is not None:
        try:
            stats_bin = get_stats(model_binary, 'elapsedyrs')
            results_obj['model_binary'] = stats_bin
            if stats_bin["p_value"] is not None and stats_bin["p_value"] < 0.05 and stats_bin["beta"] > 0:
                descriptions.append(
                    "model_binary: 'elapsedyrs' (binary female-name indicator) has a positive, statistically "
                    "significant coefficient (p < 0.05). This supports the hypothesis that female names are "
                    "associated with more deaths."
                )
            elif stats_bin["p_value"] is not None and stats_bin["p_value"] < 0.05 and stats_bin["beta"] <= 0:
                descriptions.append(
                    "model_binary: 'elapsedyrs' is statistically significant but the sign is non-positive, which "
                    "does NOT support the hypothesis."
                )
            else:
                descriptions.append(
                    "model_binary: 'elapsedyrs' is not statistically significant (p >= 0.05). No reliable evidence "
                    "from this model that binary female-name status affects log(deaths)."
                )
        except Exception as e:
            results_obj['model_binary_error'] = str(e)
            descriptions.append(f"model_binary: could not extract stats: {e}")

    # Combine interpretation into a short paragraph
    combined_description = " ".join(descriptions) if descriptions else "No models found or no interpretation available."

    return {
        "object": results_obj,
        "description": combined_description
    }