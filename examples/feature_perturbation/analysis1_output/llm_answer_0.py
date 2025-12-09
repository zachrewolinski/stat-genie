def extract_final_answer(model_output):
    """
    Extracts statistics for the 'masfem_z' coefficient from the provided model_output dict,
    which is expected to contain 'nb_result' (Negative Binomial GLM fit) and optionally
    'ols_result' (OLS on log(alldeaths + 1)).

    Returns:
      {
        "object": {
          "nb": { "coef": ..., "pvalue": ..., "ci_lower": ..., "ci_upper": ..., "irr": ..., "irr_ci_lower": ..., "irr_ci_upper": ... }  # if nb_result present
          "ols": { "coef": ..., "pvalue": ..., "ci_lower": ..., "ci_upper": ..., "pct_change": ..., "pct_change_ci_lower": ..., "pct_change_ci_upper": ... }  # if ols_result present
          "conclusion": "supports / does not support / inconclusive"  # simple judgement based on sign and p-value (alpha=0.05)
        },
        "description": "Plain-language explanation"
      }
    """
    import numpy as np

    result = {"object": {}, "description": ""}

    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dictionary containing model result objects.")

    # Helper to safely extract param, pvalue, conf int
    def _extract_from_result(res, name):
        out = {}
        # params and pvalues are usually pandas Series
        try:
            coef = float(res.params[name])
        except Exception:
            # try positional access
            try:
                idx = list(res.params.index).index(name)
                coef = float(res.params.iloc[idx])
            except Exception as e:
                raise KeyError(f"Could not find parameter '{name}' in result.params: {e}")

        # p-value
        try:
            pval = float(res.pvalues[name])
        except Exception:
            try:
                idx = list(res.pvalues.index).index(name)
                pval = float(res.pvalues.iloc[idx])
            except Exception as e:
                pval = np.nan

        # confidence interval
        try:
            ci = res.conf_int()  # may be ndarray or DataFrame
            # If DataFrame-like and has index:
            if hasattr(ci, "loc") and name in ci.index:
                lower, upper = float(ci.loc[name, 0]), float(ci.loc[name, 1])
            else:
                # assume array with ordering same as params
                idx = list(res.params.index).index(name)
                lower, upper = float(ci[idx, 0]), float(ci[idx, 1])
        except Exception:
            lower, upper = np.nan, np.nan

        out["coef"] = coef
        out["pvalue"] = pval
        out["ci_lower"] = lower
        out["ci_upper"] = upper
        return out

    # Primary: Negative Binomial
    if "nb_result" in model_output and model_output["nb_result"] is not None:
        nb = model_output["nb_result"]
        try:
            nb_stats = _extract_from_result(nb, "masfem_z")
            # For count model, exponentiate coef to get incidence rate ratio (IRR)
            irr = np.exp(nb_stats["coef"])
            irr_ci_lower = np.exp(nb_stats["ci_lower"]) if not np.isnan(nb_stats["ci_lower"]) else np.nan
            irr_ci_upper = np.exp(nb_stats["ci_upper"]) if not np.isnan(nb_stats["ci_upper"]) else np.nan

            nb_stats.update({
                "irr": irr,
                "irr_ci_lower": irr_ci_lower,
                "irr_ci_upper": irr_ci_upper
            })
            result["object"]["nb"] = nb_stats
        except KeyError as e:
            result["object"]["nb_error"] = str(e)
    else:
        result["object"]["nb"] = None

    # Robustness: OLS on log(alldeaths + 1)
    if "ols_result" in model_output and model_output["ols_result"] is not None:
        ols = model_output["ols_result"]
        try:
            ols_stats = _extract_from_result(ols, "masfem_z")
            # For log outcome, approximate percent change = (exp(coef) - 1) * 100
            pct_change = (np.exp(ols_stats["coef"]) - 1.0) * 100.0
            pct_ci_lower = (np.exp(ols_stats["ci_lower"]) - 1.0) * 100.0 if not np.isnan(ols_stats["ci_lower"]) else np.nan
            pct_ci_upper = (np.exp(ols_stats["ci_upper"]) - 1.0) * 100.0 if not np.isnan(ols_stats["ci_upper"]) else np.nan

            ols_stats.update({
                "pct_change": pct_change,
                "pct_change_ci_lower": pct_ci_lower,
                "pct_change_ci_upper": pct_ci_upper
            })
            result["object"]["ols"] = ols_stats
        except KeyError as e:
            result["object"]["ols_error"] = str(e)
    else:
        result["object"]["ols"] = None

    # Simple conclusion logic based on primary NB model
    conclusion = "inconclusive"
    try:
        nb_obj = result["object"].get("nb")
        if isinstance(nb_obj, dict) and "coef" in nb_obj:
            coef = nb_obj["coef"]
            p = nb_obj["pvalue"]
            # Direction: positive coef => more feminine names associated with more deaths (consistent with hypothesis)
            if (not np.isnan(p)) and (p <= 0.05):
                conclusion = "supports" if coef > 0 else "contradicts"
            else:
                # not statistically significant
                conclusion = "inconclusive"
        else:
            conclusion = "inconclusive"
    except Exception:
        conclusion = "inconclusive"

    # Build a readable description
    descr_lines = []
    descr_lines.append("Extracted estimates for the predictor 'masfem_z' (higher = more feminine name).")
    if result["object"].get("nb"):
        nb = result["object"]["nb"]
        descr_lines.append(
            f"Negative Binomial (primary): coef = {nb['coef']:.4f}, p = {nb['pvalue']:.4g}, "
            f"95% CI = [{nb['ci_lower']:.4f}, {nb['ci_upper']:.4f}]."
        )
        descr_lines.append(
            f"This implies an incidence rate ratio (IRR) = {nb['irr']:.4f} "
            f"(95% CI = [{nb['irr_ci_lower']:.4f}, {nb['irr_ci_upper']:.4f}])."
        )
    else:
        descr_lines.append("Negative Binomial result not available or 'masfem_z' not estimated.")

    if result["object"].get("ols"):
        ols = result["object"]["ols"]
        descr_lines.append(
            f"OLS on log(alldeaths+1) (robust): coef = {ols['coef']:.4f}, p = {ols['pvalue']:.4g}, "
            f"95% CI = [{ols['ci_lower']:.4f}, {ols['ci_upper']:.4f}]."
        )
        descr_lines.append(
            f"Approximate percent change in (alldeaths+1) per 1 SD increase in masculinity->femininity = "
            f"{ols['pct_change']:.2f}% (95% CI = [{ols['pct_change_ci_lower']:.2f}%, {ols['pct_change_ci_upper']:.2f}%])."
        )
    else:
        descr_lines.append("OLS robustness result not available or 'masfem_z' not estimated.")

    # Interpret direction relative to hypothesis
    if conclusion == "supports":
        descr_lines.append(
            "Interpretation: The NB estimate is positive and statistically significant at alpha=0.05, "
            "meaning storms with more feminine names are associated with higher fatality counts, "
            "which is consistent with the hypothesis that feminine names lead to fewer precautions and thus more deaths."
        )
    elif conclusion == "contradicts":
        descr_lines.append(
            "Interpretation: The NB estimate is negative and statistically significant at alpha=0.05, "
            "meaning storms with more feminine names are associated with fewer fatalities, "
            "which contradicts the stated hypothesis."
        )
    else:
        descr_lines.append(
            "Interpretation: The NB estimate is not statistically significant (or not available), "
            "so there is insufficient evidence to conclude that perceived femininity of hurricane names "
            "affects fatality counts in the data at conventional significance levels."
        )

    result["object"]["conclusion"] = conclusion
    result["description"] = " ".join(descr_lines)

    return result