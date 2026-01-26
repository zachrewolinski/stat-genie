def extract_final_answer(model_output):
    """
    Extract key statistics relevant to the question:
      "How do children's reliance on majority preference develop with age across cultures?"
    from the model_output dict produced by the modeling function.

    Returns a dictionary with:
      - "object": dict with numeric outputs (LR tests, age coefficients, CIs, odds-ratios, p-values, sample size)
      - "description": human-readable summary interpreting those outputs in context
    """
    import numpy as np

    out = {"object": {}, "description": ""}

    # Defensive retrieval of available pieces
    lr_interactions = model_output.get("lr_test_interactions", None)
    lr_culture = model_output.get("test_culture_main_effects", None)
    summary = model_output.get("summary", {})
    n_obs = summary.get("n_obs", None)

    out["object"]["n_obs"] = n_obs
    out["object"]["lr_interactions"] = lr_interactions
    out["object"]["lr_culture_main_effects"] = lr_culture

    # Extract multinomial reduced model coefficients for age effect (common age effect across cultures)
    reduced_res = model_output.get("reduced_result", None)
    age_effects = {}

    if reduced_res is not None:
        try:
            params = reduced_res.params            # DataFrame: rows = exog names, cols = outcome levels (excluding base)
            pvals = reduced_res.pvalues
            bse = reduced_res.bse

            # Expect an exogenous row named 'age_center'
            if "age_center" not in params.index:
                # try alternative capitalization/spaces (defensive)
                possible_age_rows = [r for r in params.index if "age" in r.lower() and "center" in r.lower()]
            else:
                possible_age_rows = ["age_center"]

            if not possible_age_rows:
                raise KeyError("Could not find an 'age_center' row in reduced model params.")

            age_row = possible_age_rows[0]

            # Iterate columns (each column corresponds to a non-reference outcome)
            for col in params.columns:
                coef = float(params.loc[age_row, col])
                se = float(bse.loc[age_row, col])
                pval = float(pvals.loc[age_row, col])
                ci_low = coef - 1.96 * se
                ci_high = coef + 1.96 * se
                or_coef = float(np.exp(coef))
                or_ci = (float(np.exp(ci_low)), float(np.exp(ci_high)))

                # Attempt to map column name to outcome label (y_cat: 0=unchosen, 1=majority, 2=minority)
                try:
                    col_int = int(str(col))
                    if col_int == 1:
                        label = "majority_option_vs_unchosen"
                    elif col_int == 2:
                        label = "minority_option_vs_unchosen"
                    else:
                        label = f"outcome_{col}"
                except Exception:
                    label = f"outcome_{col}"

                age_effects[label] = {
                    "coef": coef,
                    "se": se,
                    "z": coef / se if se != 0 else None,
                    "p_value": pval,
                    "ci_95": (ci_low, ci_high),
                    "odds_ratio": or_coef,
                    "odds_ratio_ci_95": or_ci,
                    "model_column": col
                }

            out["object"]["age_effects_reduced_model"] = age_effects

        except Exception as e:
            out["object"]["age_effects_reduced_model_error"] = str(e)
    else:
        out["object"]["age_effects_reduced_model"] = None

    # Build human-readable description based on available numbers
    desc_lines = []
    desc_lines.append(f"Sample size: {n_obs} observations." if n_obs is not None else "Sample size: unknown.")

    # Interpret LR test for age-by-culture interactions
    if lr_interactions is not None:
        p_int = lr_interactions.get("p_value", None)
        stat_int = lr_interactions.get("lr_stat", None)
        df_int = lr_interactions.get("df_diff", None)
        desc_lines.append(
            "Likelihood-ratio test for age-by-culture interactions: "
            f"LR stat = {stat_int}, df = {df_int}, p = {p_int}."
        )
        if p_int is not None:
            if p_int < 0.05:
                desc_lines.append("This suggests evidence that developmental trajectories (age effects) differ across cultures.")
            else:
                desc_lines.append(
                    "This does NOT provide statistically significant evidence that age effects differ across cultures "
                    "(p >= 0.05). Interpreting age effects as common across cultures is reasonable."
                )
    else:
        desc_lines.append("No LR test for age-by-culture interactions available.")

    # Interpret culture main effects test
    if lr_culture is not None:
        p_c = lr_culture.get("p_value", None)
        stat_c = lr_culture.get("lr_stat", None)
        df_c = lr_culture.get("df_diff", None)
        desc_lines.append(
            "Likelihood-ratio test for culture main effects (differences in intercepts): "
            f"LR stat = {stat_c}, df = {df_c}, p = {p_c}."
        )
        if p_c is not None:
            if p_c < 0.05:
                desc_lines.append("This suggests baseline choice propensities differ across cultural sites.")
            else:
                desc_lines.append("No statistically significant baseline differences across cultures were detected (p >= 0.05).")
    else:
        desc_lines.append("No LR test for culture main effects available.")

    # Summarize age coefficients per outcome
    if age_effects:
        for label, stats_dict in age_effects.items():
            coef = stats_dict["coef"]
            pval = stats_dict["p_value"]
            orr = stats_dict["odds_ratio"]
            ci = stats_dict["ci_95"]
            or_ci = stats_dict["odds_ratio_ci_95"]

            # Interpret sign and significance
            if pval < 0.05:
                sig_text = "statistically significant"
            else:
                sig_text = "not statistically significant"

            # Direction interpretation: positive coef -> higher log-odds (and OR>1) with increasing age
            if coef > 0:
                direction = "increases with age (positive association)"
            elif coef < 0:
                direction = "decreases with age (negative association)"
            else:
                direction = "no change with age (coef = 0)"

            # Map label to plain-language summary
            if label == "majority_option_vs_unchosen":
                plain = "Choosing the majority demonstrated option (vs unchosen)"
            elif label == "minority_option_vs_unchosen":
                plain = "Choosing the minority demonstrated option (vs unchosen)"
            else:
                plain = label

            desc_lines.append(
                f"{plain}: coefficient = {coef:.4f}, SE = {stats_dict['se']:.4f}, 95% CI = ({ci[0]:.4f}, {ci[1]:.4f}), "
                f"OR = {orr:.3f} (95% CI = {or_ci[0]:.3f}–{or_ci[1]:.3f}), p = {pval:.3f}. "
                f"Interpretation: {direction}; this effect is {sig_text}."
            )
    else:
        desc_lines.append("Age effect estimates are not available from the reduced multinomial model output.")

    out["description"] = " ".join(desc_lines)

    return out