def extract_final_answer(model_output):
    """
    Extracts relevant statistics about the effect of 'femininity_z' on log fatalities
    (primary outcome), and also from the interaction and damage robustness models if present.

    Returns a dict with:
      - "object": a nested dict containing numeric results for main, interaction, and damage models
      - "description": a concise human-readable interpretation and a yes/no assessment
                       of whether the results support the hypothesis that more feminine
                       hurricane names lead to higher fatalities/damage (via lower precautions).
    """
    results = {}
    summary_lines = []

    def extract_from_model(res, term):
        """
        Given a statsmodels RegressionResultsWrapper `res` and a parameter name `term`,
        return a dict with coefficient, se, pvalue, 95% CI, nobs, and rsquared (if available).
        If term not present, return None.
        """
        if res is None:
            return None
        params = getattr(res, "params", None)
        if params is None or term not in params.index:
            return None
        coef = float(params[term])
        se = float(res.bse[term]) if hasattr(res, "bse") and term in res.bse.index else None
        pval = float(res.pvalues[term]) if hasattr(res, "pvalues") and term in res.pvalues.index else None
        try:
            ci = res.conf_int().loc[term].astype(float).tolist()
            ci_lower, ci_upper = float(ci[0]), float(ci[1])
        except Exception:
            ci_lower, ci_upper = None, None
        nobs = int(res.nobs) if hasattr(res, "nobs") else None
        rsq = float(res.rsquared) if hasattr(res, "rsquared") else None
        return {
            "term": term,
            "coef": coef,
            "se": se,
            "pvalue": pval,
            "ci_lower_95": ci_lower,
            "ci_upper_95": ci_upper,
            "nobs": nobs,
            "rsquared": rsq
        }

    # Main model (deaths)
    deaths_model = model_output.get("deaths_model") if isinstance(model_output, dict) else None
    main_stats = extract_from_model(deaths_model, "femininity_z")
    results["main_model"] = main_stats
    if main_stats is None:
        summary_lines.append("Main model: 'femininity_z' not found or model missing.")
    else:
        sign = "positive" if main_stats["coef"] > 0 else ("negative" if main_stats["coef"] < 0 else "zero")
        signif = (main_stats["pvalue"] is not None and main_stats["pvalue"] < 0.05)
        summary_lines.append(
            "Main model (log fatalities): femininity_z coef = {coef:.4f}, SE = {se:.4f}, p = {p:.4g}, "
            "95% CI = [{lo:.4f}, {hi:.4f}], n = {n}, R2 = {r:.3f}. "
            "Coefficient is {sign}.".format(
                coef=main_stats["coef"],
                se=main_stats["se"] if main_stats["se"] is not None else float("nan"),
                p=main_stats["pvalue"] if main_stats["pvalue"] is not None else float("nan"),
                lo=main_stats["ci_lower_95"] if main_stats["ci_lower_95"] is not None else float("nan"),
                hi=main_stats["ci_upper_95"] if main_stats["ci_upper_95"] is not None else float("nan"),
                n=main_stats["nobs"],
                r=main_stats["rsquared"] if main_stats["rsquared"] is not None else 0.0,
                sign=sign
            )
        )
        # Decide whether this supports the hypothesis:
        # Hypothesis predicts higher femininity -> higher fatalities (positive coef) and statistically significant.
        supports = (main_stats["coef"] > 0) and (main_stats["pvalue"] is not None and main_stats["pvalue"] < 0.05)
        summary_lines.append("Does the main model support the hypothesis? {}".format("Yes" if supports else "No"))

    # Interaction model
    interaction_model = model_output.get("interaction_model") if isinstance(model_output, dict) else None
    interact_stats_fem = extract_from_model(interaction_model, "femininity_z")
    interact_stats_term = extract_from_model(interaction_model, "fem_x_wind")
    results["interaction_model"] = {
        "femininity_main": interact_stats_fem,
        "fem_x_wind": interact_stats_term
    }
    if interaction_model is None:
        summary_lines.append("Interaction model: not provided.")
    else:
        if interact_stats_fem is None and interact_stats_term is None:
            summary_lines.append("Interaction model: neither 'femininity_z' nor 'fem_x_wind' found.")
        else:
            if interact_stats_fem is not None:
                summary_lines.append(
                    "Interaction model (main fem term): coef = {coef:.4f}, p = {p:.4g}.".format(
                        coef=interact_stats_fem["coef"],
                        p=interact_stats_fem["pvalue"] if interact_stats_fem["pvalue"] is not None else float("nan")
                    )
                )
            if interact_stats_term is not None:
                signif_int = (interact_stats_term["pvalue"] is not None and interact_stats_term["pvalue"] < 0.05)
                direction = "positive" if interact_stats_term["coef"] > 0 else ("negative" if interact_stats_term["coef"] < 0 else "zero")
                summary_lines.append(
                    "Interaction term (fem_x_wind): coef = {coef:.4f}, SE = {se:.4f}, p = {p:.4g}. "
                    "Interaction is {dir}{sig}.".format(
                        coef=interact_stats_term["coef"],
                        se=interact_stats_term["se"] if interact_stats_term["se"] is not None else float("nan"),
                        p=interact_stats_term["pvalue"] if interact_stats_term["pvalue"] is not None else float("nan"),
                        dir=direction,
                        sig=(" and statistically significant" if signif_int else " but not statistically significant")
                    )
                )
                if signif_int:
                    summary_lines.append("Interpretation: the effect of perceived femininity on fatalities depends on wind (storm intensity).")

    # Damage robustness model
    damage_model = model_output.get("damage_model") if isinstance(model_output, dict) else None
    damage_stats = extract_from_model(damage_model, "femininity_z")
    results["damage_model"] = damage_stats
    if damage_stats is None:
        summary_lines.append("Damage model: 'femininity_z' not found or damage model missing.")
    else:
        sign = "positive" if damage_stats["coef"] > 0 else ("negative" if damage_stats["coef"] < 0 else "zero")
        signif = (damage_stats["pvalue"] is not None and damage_stats["pvalue"] < 0.05)
        summary_lines.append(
            "Damage model (log property damage): femininity_z coef = {coef:.4f}, SE = {se:.4f}, p = {p:.4g}, "
            "95% CI = [{lo:.4f}, {hi:.4f}], n = {n}.".format(
                coef=damage_stats["coef"],
                se=damage_stats["se"] if damage_stats["se"] is not None else float("nan"),
                p=damage_stats["pvalue"] if damage_stats["pvalue"] is not None else float("nan"),
                lo=damage_stats["ci_lower_95"] if damage_stats["ci_lower_95"] is not None else float("nan"),
                hi=damage_stats["ci_upper_95"] if damage_stats["ci_upper_95"] is not None else float("nan"),
                n=damage_stats["nobs"]
            )
        )
        summary_lines.append("Does the damage model support the hypothesis? {}".format(
            "Yes" if (damage_stats["coef"] > 0 and damage_stats["pvalue"] is not None and damage_stats["pvalue"] < 0.05) else "No"
        ))

    description = " ".join(summary_lines)

    return {"object": results, "description": description}