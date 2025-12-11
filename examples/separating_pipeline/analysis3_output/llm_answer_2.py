def extract_final_answer(model_output):
    """
    Extracts coefficient, SE, t-stat, p-value, 95% CI, and sample size for the predictor
    'FemininityIndex' from a statsmodels RegressionResultsWrapper object.

    Returns a dictionary with:
      - "object": a dict with numeric results (coef, se, t, p, ci_lower, ci_upper, n_obs, predictor)
      - "description": a short plain-language interpretation in the context of the task.

    If 'FemininityIndex' is not present in the model, returns an object=None and an explanatory description.
    """
    result = {"object": None, "description": ""}

    try:
        res = model_output  # expected to be a statsmodels RegressionResultsWrapper

        predictor = "FemininityIndex"
        params = getattr(res, "params", None)
        if params is None or predictor not in params.index:
            result["description"] = (
                f"Predictor '{predictor}' not found in the provided model output. "
                "No statistics extracted."
            )
            return result

        coef = float(params[predictor])
        bse = float(res.bse[predictor]) if hasattr(res, "bse") and predictor in res.bse.index else None
        tstat = float(res.tvalues[predictor]) if hasattr(res, "tvalues") and predictor in res.tvalues.index else None
        pval = float(res.pvalues[predictor]) if hasattr(res, "pvalues") and predictor in res.pvalues.index else None

        # Confidence interval (95%)
        try:
            ci_all = res.conf_int()  # DataFrame or ndarray-like
            # If it's a DataFrame, use .loc; otherwise index into rows
            if hasattr(ci_all, "loc"):
                ci_lower, ci_upper = float(ci_all.loc[predictor, 0]), float(ci_all.loc[predictor, 1])
            else:
                # assume order matches params.index
                idx = list(params.index).index(predictor)
                ci_lower, ci_upper = float(ci_all[idx, 0]), float(ci_all[idx, 1])
        except Exception:
            ci_lower, ci_upper = None, None

        # Sample size: try common attributes, fallback to model_data if attached by the caller
        n_obs = None
        if hasattr(res, "nobs"):
            try:
                n_obs = int(res.nobs)
            except Exception:
                n_obs = None
        if n_obs is None and hasattr(res, "model_data"):
            try:
                n_obs = int(res.model_data.get("n_obs", None))
            except Exception:
                n_obs = None

        stats = {
            "predictor": predictor,
            "coef": coef,
            "std_error": bse,
            "t_stat": tstat,
            "p_value": pval,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "n_obs": n_obs,
        }

        # Construct interpretation: coefficient is in units of logged deaths (log(ndam15 + 1))
        # Positive coef => higher femininity -> higher logged fatalities (consistent with hypothesis that feminine names lead to fewer precautions)
        significance = ""
        if pval is not None:
            if pval < 0.001:
                significance = "p < 0.001 (highly significant)"
            elif pval < 0.01:
                significance = "p < 0.01 (very significant)"
            elif pval < 0.05:
                significance = "p < 0.05 (statistically significant)"
            else:
                significance = f"p = {pval:.3f} (not statistically significant at conventional levels)"

        descr_lines = [
            f"Extracted result for predictor '{predictor}': coefficient = {coef:.4f}.",
            f"Standard error = {bse:.4f}" if bse is not None else "Standard error not available.",
            f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]" if ci_lower is not None and ci_upper is not None else "95% CI not available.",
            f"t-statistic = {tstat:.3f}" if tstat is not None else "",
            f"{significance}",
            f"Number of observations = {n_obs}" if n_obs is not None else "Sample size (n) not available.",
            "",
            "Interpretation: the coefficient is the expected change in LogDeaths (log(ndam15 + 1))",
            "for a one-unit increase in the FemininityIndex, holding controls constant.",
        ]
        # Add hypothesis-specific sentence
        if pval is not None and coef is not None:
            if coef > 0 and pval < 0.05:
                descr_lines.append(
                    "This positive and statistically significant coefficient is consistent with the hypothesis "
                    "that hurricanes with more feminine names are associated with higher fatalities (which would be "
                    "consistent with fewer precautionary measures)."
                )
            elif coef > 0 and pval >= 0.05:
                descr_lines.append(
                    "The coefficient is positive (suggesting more feminine names associate with higher fatalities), "
                    "but it is not statistically significant, so the evidence is weak."
                )
            elif coef < 0 and pval < 0.05:
                descr_lines.append(
                    "This negative and statistically significant coefficient runs counter to the hypothesis: more feminine "
                    "names are associated with lower fatalities."
                )
            else:
                descr_lines.append(
                    "The coefficient is negative (suggesting more feminine names associate with lower fatalities), "
                    "but it is not statistically significant."
                )

        result["object"] = stats
        result["description"] = " ".join([ln for ln in descr_lines if ln])  # join non-empty lines

        return result

    except Exception as e:
        result["description"] = f"An error occurred while extracting statistics: {e}"
        return result