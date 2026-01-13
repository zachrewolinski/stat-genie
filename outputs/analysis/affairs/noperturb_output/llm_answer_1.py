def extract_final_answer(model_output):
    """
    Extracts the effect of 'HasChildren' from the provided model_output dict.
    Expects model_output to contain keys 'logit_result' and 'nb_result' with
    fitted statsmodels results objects (BinaryResultsWrapper / GLMResultsWrapper).

    Returns:
      {
        "object": {
          "logit": {
            "coef": float,
            "se": float,
            "pvalue": float,
            "ci_lower": float,
            "ci_upper": float,
            "odds_ratio": float,
            "or_ci_lower": float,
            "or_ci_upper": float,
            "significant": bool
          },
          "nb": {
            "coef": float,
            "se": float,
            "pvalue": float,
            "ci_lower": float,
            "ci_upper": float,
            "irr": float,
            "irr_ci_lower": float,
            "irr_ci_upper": float,
            "significant": bool
          }
        },
        "description": str
      }
    The description summarizes whether having children appears to decrease
    engagement in extramarital affairs, based on coefficient signs and
    significance in both models.
    """
    import numpy as np

    def _get_param_info(res, param_name):
        # Extract coef, se, pvalue, conf int robustly
        if param_name not in res.params.index:
            raise KeyError(f"Parameter '{param_name}' not found in model results.")
        coef = float(res.params[param_name])
        se = float(res.bse[param_name]) if hasattr(res, 'bse') else float(np.nan)
        pval = float(res.pvalues[param_name]) if hasattr(res, 'pvalues') else float(np.nan)

        # conf_int may return array or DataFrame; handle both
        try:
            ci = res.conf_int().loc[param_name].astype(float)
            ci_lower = float(ci.iloc[0])
            ci_upper = float(ci.iloc[1])
        except Exception:
            # fallback: use position of param
            try:
                ci_arr = res.conf_int()
                idx = list(res.params.index).index(param_name)
                ci_lower = float(ci_arr[idx, 0])
                ci_upper = float(ci_arr[idx, 1])
            except Exception:
                ci_lower = float(np.nan)
                ci_upper = float(np.nan)

        return {
            "coef": coef,
            "se": se,
            "pvalue": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        }

    # Validate input
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict with keys 'logit_result' and 'nb_result'.")

    if 'logit_result' not in model_output or 'nb_result' not in model_output:
        raise KeyError("model_output must contain 'logit_result' and 'nb_result' keys.")

    logit_res = model_output['logit_result']
    nb_res = model_output['nb_result']

    # Extract parameter info for HasChildren from both models
    try:
        logit_info = _get_param_info(logit_res, 'HasChildren')
    except KeyError as e:
        raise KeyError("Failed to extract 'HasChildren' from logit_result: " + str(e))
    try:
        nb_info = _get_param_info(nb_res, 'HasChildren')
    except KeyError as e:
        raise KeyError("Failed to extract 'HasChildren' from nb_result: " + str(e))

    # Compute odds ratio and IRR and their CIs
    logit_or = float(np.exp(logit_info["coef"]))
    logit_or_ci_lower = float(np.exp(logit_info["ci_lower"])) if not np.isnan(logit_info["ci_lower"]) else float(np.nan)
    logit_or_ci_upper = float(np.exp(logit_info["ci_upper"])) if not np.isnan(logit_info["ci_upper"]) else float(np.nan)

    nb_irr = float(np.exp(nb_info["coef"]))
    nb_irr_ci_lower = float(np.exp(nb_info["ci_lower"])) if not np.isnan(nb_info["ci_lower"]) else float(np.nan)
    nb_irr_ci_upper = float(np.exp(nb_info["ci_upper"])) if not np.isnan(nb_info["ci_upper"]) else float(np.nan)

    # Determine significance at alpha=0.05
    logit_sig = (not np.isnan(logit_info["pvalue"])) and (logit_info["pvalue"] < 0.05)
    nb_sig = (not np.isnan(nb_info["pvalue"])) and (nb_info["pvalue"] < 0.05)

    # Prepare object to return
    output_object = {
        "logit": {
            "coef": logit_info["coef"],
            "se": logit_info["se"],
            "pvalue": logit_info["pvalue"],
            "ci_lower": logit_info["ci_lower"],
            "ci_upper": logit_info["ci_upper"],
            "odds_ratio": logit_or,
            "or_ci_lower": logit_or_ci_lower,
            "or_ci_upper": logit_or_ci_upper,
            "significant": logit_sig
        },
        "nb": {
            "coef": nb_info["coef"],
            "se": nb_info["se"],
            "pvalue": nb_info["pvalue"],
            "ci_lower": nb_info["ci_lower"],
            "ci_upper": nb_info["ci_upper"],
            "irr": nb_irr,
            "irr_ci_lower": nb_irr_ci_lower,
            "irr_ci_upper": nb_irr_ci_upper,
            "significant": nb_sig
        }
    }

    # Construct a concise human-readable description
    parts = []
    # Logit interpretation
    parts.append(
        "Logistic model (AnyAffair): coef={coef:.4f}, SE={se:.4f}, p={p:.4g}; "
        "odds ratio={or_: .4f} (95% CI [{or_lo:.4f}, {or_hi:.4f}]).".format(
            coef=logit_info["coef"], se=logit_info["se"], p=logit_info["pvalue"],
            or_=logit_or, or_lo=logit_or_ci_lower, or_hi=logit_or_ci_upper
        )
    )
    if logit_sig:
        if logit_or < 1:
            parts.append("Interpretation: having children is associated with statistically significant lower odds of any extramarital affair (logit).")
        else:
            parts.append("Interpretation: having children is associated with statistically significant higher odds of any extramarital affair (logit).")
    else:
        parts.append("Interpretation: effect not statistically significant in the logistic model (p >= 0.05).")

    # NB interpretation
    parts.append(
        "Negative binomial (count of Affairs): coef={coef:.4f}, SE={se:.4f}, p={p:.4g}; "
        "IRR={irr:.4f} (95% CI [{irr_lo:.4f}, {irr_hi:.4f}]).".format(
            coef=nb_info["coef"], se=nb_info["se"], p=nb_info["pvalue"],
            irr=nb_irr, irr_lo=nb_irr_ci_lower, irr_hi=nb_irr_ci_upper
        )
    )
    if nb_sig:
        if nb_irr < 1:
            parts.append("Interpretation: having children is associated with a statistically significant lower expected count of affairs (negative binomial).")
        else:
            parts.append("Interpretation: having children is associated with a statistically significant higher expected count of affairs (negative binomial).")
    else:
        parts.append("Interpretation: effect not statistically significant in the negative binomial model (p >= 0.05).")

    # Overall summary
    if logit_sig and nb_sig and (logit_or < 1) and (nb_irr < 1):
        summary = "Overall: Both models suggest having children decreases engagement in extramarital affairs (statistically significant)."
    elif (logit_sig and (logit_or < 1)) or (nb_sig and (nb_irr < 1)):
        summary = "Overall: Evidence is mixed but at least one model shows a statistically significant decrease in affairs associated with having children."
    else:
        summary = "Overall: There is no consistent statistically significant evidence that having children decreases engagement in extramarital affairs."

    parts.append(summary)

    description = " ".join(parts)

    return {"object": output_object, "description": description}