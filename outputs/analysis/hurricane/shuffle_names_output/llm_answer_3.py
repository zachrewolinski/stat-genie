def extract_final_answer(model_output):
    """
    Extract key statistics for the coefficient on 'name_mf_z' from the provided
    model_output dictionary (expects keys 'ols_robust' and optionally 'neg_binom').

    Returns:
      {
        "object": {
          "ols": {
            "coef": float,
            "std_err": float,
            "p_value": float,
            "ci_95": (float, float),
            "percent_change_in_1_plus_deaths": float   # (exp(coef)-1)*100
          } or None,
          "neg_binom": {
            "coef": float,
            "std_err": float,
            "p_value": float,
            "ci_95": (float, float),
            "percent_change_in_expected_count": float  # (exp(coef)-1)*100
          } or None
        },
        "description": str  # short interpretation in context
      }
    """
    import numpy as np

    def _safe_get_stats(res, param_name):
        """Return dict of stats for param_name from a statsmodels results object or None."""
        if res is None:
            return None
        # Some results objects expose params as an Index; check presence
        try:
            params_index = list(res.params.index)
        except Exception:
            # If params attribute not available or not indexable, give up
            return None
        if param_name not in params_index:
            return None
        try:
            coef = float(res.params[param_name])
        except Exception:
            coef = None
        # bse and pvalues might be present
        try:
            se = float(res.bse[param_name])
        except Exception:
            se = None
        try:
            pval = float(res.pvalues[param_name])
        except Exception:
            pval = None
        # conf_int might be an array-like or DataFrame; handle robustly
        try:
            ci_all = res.conf_int()
            # If it's a DataFrame-like, try .loc
            try:
                lower, upper = ci_all.loc[param_name].tolist()
            except Exception:
                # fallback: find row index
                try:
                    row_idx = params_index.index(param_name)
                    lower, upper = ci_all.iloc[row_idx].tolist()
                except Exception:
                    # fallback: if ci_all is ndarray and columns correspond to params order
                    try:
                        row_idx = params_index.index(param_name)
                        lower, upper = float(ci_all[row_idx, 0]), float(ci_all[row_idx, 1])
                    except Exception:
                        lower, upper = None, None
        except Exception:
            lower, upper = None, None

        # Compute percentage changes based on link function:
        # For log-transformed dependent variable (OLS): percent change in (1 + deaths) ~ (exp(coef)-1)*100
        # For count log-link models (NegativeBinomial GLM): percent change in expected count ~ (exp(coef)-1)*100
        pct_change = None
        if coef is not None:
            try:
                pct_change = (np.exp(coef) - 1.0) * 100.0
            except Exception:
                pct_change = None

        return {
            "coef": coef,
            "std_err": se,
            "p_value": pval,
            "ci_95": (lower, upper),
            "percent_change": pct_change
        }

    # Extract models from input dict
    ols_res = model_output.get("ols_robust")
    nb_res = model_output.get("neg_binom")

    ols_stats = _safe_get_stats(ols_res, "name_mf_z")
    nb_stats = _safe_get_stats(nb_res, "name_mf_z")

    # Build interpretation text
    lines = []
    lines.append("Inference for effect of 'name_mf_z' (higher = more feminine):")

    def interpret_block(stats, label, outcome_desc):
        if stats is None:
            return f"- {label}: model or parameter not available."
        coef = stats["coef"]
        p = stats["p_value"]
        ci = stats["ci_95"]
        pct = stats["percent_change"]
        # Decide significance at conventional 0.05
        sig_text = "statistically significant (p < 0.05)" if (p is not None and p < 0.05) else \
                   ("marginal (0.05 <= p < 0.10)" if (p is not None and p < 0.10) else "not statistically significant")
        # Build summary
        s = (f"- {label}: coef = {coef:.4f}" if coef is not None else f"- {label}: coef = NA")
        if p is not None:
            s += f", p = {p:.3f}"
        else:
            s += ", p = NA"
        if ci[0] is not None and ci[1] is not None:
            s += f", 95% CI = [{ci[0]:.4f}, {ci[1]:.4f}]"
        if pct is not None:
            s += f". Interpreted on the {outcome_desc}, this corresponds to a {pct:.1f}% change per one-unit increase in name_mf_z (one SD since name_mf_z is z-scored)."
        s += f" ({sig_text})."
        return s

    # OLS: outcome is log(1 + deaths) -> percent change in (1 + deaths)
    lines.append(interpret_block(ols_stats, "OLS (robust HC3) on log_deaths", "log(1 + deaths)"))
    # Negative binomial: outcome is raw counts -> percent change in expected count
    lines.append(interpret_block(nb_stats, "Negative Binomial (GLM, log link) on ndam15", "expected count"))

    # Concluding statement about hypothesis
    conclusion = ""
    if ols_stats is not None and ols_stats["p_value"] is not None:
        if ols_stats["p_value"] < 0.05:
            # direction matters
            if ols_stats["coef"] > 0:
                conclusion = ("Conclusion (OLS): Evidence supports the hypothesis that more feminine hurricane names "
                              "are associated with higher fatalities (fewer effective precautions), "
                              "statistically significant at p < 0.05.")
            else:
                conclusion = ("Conclusion (OLS): Evidence indicates more feminine hurricane names are associated with LOWER fatalities, "
                              "contrary to the hypothesis, statistically significant at p < 0.05.")
        else:
            conclusion = ("Conclusion (OLS): No statistically significant evidence (p >= 0.05) that name femininity affects fatalities.")
    else:
        conclusion = "Conclusion: Could not determine statistical significance from OLS results."

    # Also mention whether NB corroborates
    if nb_stats is not None and nb_stats["p_value"] is not None:
        if nb_stats["p_value"] < 0.05:
            nb_statement = ("Negative binomial model shows a statistically significant effect in the same direction."
                            if (ols_stats and nb_stats["coef"] is not None and ols_stats["coef"] is not None and
                                np.sign(nb_stats["coef"]) == np.sign(ols_stats["coef"]))
                            else "Negative binomial model shows a statistically significant effect (potentially different direction).")
        else:
            nb_statement = "Negative binomial model does not show a statistically significant effect (p >= 0.05)."
        conclusion += " " + nb_statement

    # Assemble return object
    result_object = {
        "ols": ols_stats,
        "neg_binom": nb_stats
    }
    description = "\n".join(lines) + "\n\n" + conclusion

    return {"object": result_object, "description": description}