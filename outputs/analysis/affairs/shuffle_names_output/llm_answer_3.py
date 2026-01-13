def extract_final_answer(model_output):
    """
    Extracts statistics for the 'HasChildren' coefficient from a fitted statsmodels results object
    (including robust cov results wrappers).

    Returns a dictionary with:
      - "object": a dict containing numeric results (log-odds coef, SE, p-value, 95% CI, odds ratio and OR CI)
      - "description": a concise interpretation in context (statistical significance and direction)

    If extraction fails, returns object=None and an error description.
    """
    import numpy as np

    try:
        res = model_output

        # Attempt to locate the parameter name exactly; fallback if index naming is unexpected
        param_name = None
        if hasattr(res, "params"):
            if "HasChildren" in res.params.index:
                param_name = "HasChildren"
            else:
                # try to find something close to HasChildren (case-insensitive or partial match)
                for nm in res.params.index:
                    if str(nm).lower() == "haschildren" or "haschildren" in str(nm).lower():
                        param_name = nm
                        break

        if param_name is None:
            raise KeyError("Could not find a parameter named 'HasChildren' in model_output.params.index")

        coef = float(res.params[param_name])
        se = float(res.bse[param_name]) if hasattr(res, "bse") else None
        pval = float(res.pvalues[param_name]) if hasattr(res, "pvalues") else None

        # 95% confidence interval on log-odds scale
        if hasattr(res, "conf_int"):
            ci_df = res.conf_int()
            # conf_int may return numpy array or DataFrame; handle both
            try:
                lower_log, upper_log = float(ci_df.loc[param_name, 0]), float(ci_df.loc[param_name, 1])
            except Exception:
                # fallback if conf_int returned an array-like with same order as params
                idx = list(res.params.index).index(param_name)
                lower_log, upper_log = float(ci_df[idx, 0]), float(ci_df[idx, 1])
        else:
            lower_log = float(coef - 1.96 * se) if se is not None else None
            upper_log = float(coef + 1.96 * se) if se is not None else None

        # Odds ratio and its CI
        odds_ratio = float(np.exp(coef))
        or_ci = [float(np.exp(lower_log)), float(np.exp(upper_log))] if (lower_log is not None and upper_log is not None) else [None, None]

        result_object = {
            "coef_log_odds": coef,
            "std_error": se,
            "p_value": pval,
            "conf_int_log_odds_95": [lower_log, upper_log],
            "odds_ratio": odds_ratio,
            "conf_int_odds_ratio_95": or_ci
        }

        # Brief interpretation
        if pval is None:
            significance_text = "p-value unavailable; cannot judge statistical significance."
        else:
            if pval < 0.01:
                sig_level = "p < 0.01"
            elif pval < 0.05:
                sig_level = "p < 0.05"
            elif pval < 0.1:
                sig_level = "p < 0.10"
            else:
                sig_level = f"p = {pval:.3f}"

            if pval < 0.05:
                direction = "decrease" if coef < 0 else "increase"
                significance_text = f"Statistically significant ({sig_level}): associated with a {direction} in odds of any extramarital affair."
            else:
                direction = "decrease" if coef < 0 else "increase"
                significance_text = f"Not statistically significant ({sig_level}): point estimate indicates a {direction} in odds, but we cannot reject no effect."

        description = (
            f"Estimated effect of having children ('HasChildren') from the logistic regression (controls: IsMale, YearsMarried, "
            f"Education, Religiousness, Age, MaritalRating).\n"
            f"Log-odds coefficient = {coef:.4f}, SE = {se:.4f if se is not None else 'NA'}, 95% CI (log-odds) = [{lower_log:.4f}, {upper_log:.4f}].\n"
            f"Odds ratio = {odds_ratio:.4f}, 95% CI (OR) = [{or_ci[0]:.4f}, {or_ci[1]:.4f}].\n"
            f"{significance_text}\n"
            f"Interpretation: this reflects an association (not proof of causation) between having children and the odds of reporting any extramarital affair in the past year."
        )

        return {"object": result_object, "description": description}

    except Exception as e:
        return {"object": None, "description": f"Error extracting HasChildren statistics: {e}"}